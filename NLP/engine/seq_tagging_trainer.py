import os
import numpy as np
import torch
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from model_args import SeqTaggingArgs
from .seq_tagging_utils import *


class TrainerForSeqTagging:
    def train(self, model, train_data, device, args, test_data=None):
        model_args = SeqTaggingArgs()

        model_args.update_from_dict(
            {
                "fl_algorithm": args.federated_optimizer,
                # "freeze_layers": args.freeze_layers,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "do_lower_case": args.do_lower_case,
                # "manual_seed": args.random_seed,
                # for ignoring the cache features.
                "reprocess_input_data": args.reprocess_input_data,
                "overwrite_output_dir": True,
                "max_seq_length": args.max_seq_length,
                "train_batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "evaluate_during_training": False,  # Disabled for FedAvg.
                "evaluate_during_training_steps": args.evaluate_during_training_steps,
                "fp16": args.fp16,
                "data_file_path": args.data_file_path,
                "partition_file_path": args.partition_file_path,
                "partition_method": args.partition_method,
                "dataset": args.dataset,
                "output_dir": args.output_dir,
                # "is_debug_mode": args.is_debug_mode,
                # "fedprox_mu": args.fedprox_mu,
                "optimizer": args.client_optimizer,
            }
        )

        model.to(device)
        model.train()
        tr_loss = 0
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        iteration_in_total = (
            len(train_data) // args.gradient_accumulation_steps * args.epochs
        )
        optimizer, scheduler = build_optimizer(model, iteration_in_total, model_args)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, batch in enumerate(train_data):
                x = batch[1].to(device)
                labels = batch[4].to(device)
                log_probs = model(x)
                log_probs = log_probs[0]
                loss = criterion(log_probs.view(-1, args.num_labels), labels.view(-1))

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                if hasattr(model, "backward_param"):
                    model.backward_param()
                tr_loss += loss.item()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.clip_grad_norm == 1:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    batch_loss.append(tr_loss)
                    tr_loss = 0

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if args.evaluate_during_training and test_data is not None:
                results, _, _ = self.test(test_data, device, args)
        return np.mean(epoch_loss)

    def test(self, model, test_data, device, args):
        attributes = load_attributes(args.data_file_path)
        args.num_labels = len(attributes["label_vocab"])
        args.labels_list = list(attributes["label_vocab"].keys())
        args.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        results = {}
        eval_loss = 0.0
        nb_eval_steps = 0

        n_batches = len(test_data)

        test_sample_len = len(test_data.dataset)
        pad_token_label_id = args.pad_token_label_id
        eval_output_dir = args.output_dir

        preds = None
        out_label_ids = None

        model.to(device)
        model.eval()

        for i, batch in enumerate(test_data):
            batch = tuple(t for t in batch)
            with torch.no_grad():
                sample_index_list = batch[0].to(device).cpu().numpy()

                x = batch[1].to(device)
                labels = batch[4].to(device)

                output = model(x)
                logits = output[0]

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1
            start_index = args.eval_batch_size * i

            end_index = (
                start_index + args.eval_batch_size
                if i != (n_batches - 1)
                else test_sample_len
            )

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[4].detach().cpu().numpy()
                out_input_ids = batch[1].detach().cpu().numpy()
                out_attention_mask = batch[2].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, batch[4].detach().cpu().numpy(), axis=0
                )
                out_input_ids = np.append(
                    out_input_ids, batch[1].detach().cpu().numpy(), axis=0
                )
                out_attention_mask = np.append(
                    out_attention_mask,
                    batch[2].detach().cpu().numpy(),
                    axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps

        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        result = {
            "test_loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1_score": f1_score(out_label_list, preds_list),
        }

        os.makedirs(eval_output_dir, exist_ok=True)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        return result


def load_attributes(data_path):
    data_file = h5py.File(data_path, "r", swmr=True)
    attributes = json.loads(data_file["attributes"][()])
    data_file.close()
    return attributes
