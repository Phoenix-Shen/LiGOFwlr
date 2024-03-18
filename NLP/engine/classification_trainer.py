import copy
import torch
from model_args import ClassificationArgs
from .text_classification_utils import *
import numpy as np


class TrainerForClassification:
    def train(self, model, train_data, device, args, test_data=None):
        model_args = ClassificationArgs()
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
        criterion = torch.nn.CrossEntropyLoss().to(device)
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
                loss = criterion(log_probs, labels)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                if hasattr(model, "backward_param"):
                    model.backward_param()
                tr_loss += loss.item()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.clip_grad_norm:
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
                metrics = self.test(test_data, device, args)

        return np.mean(epoch_loss)

    def test(self, model, test_data, device, args):
        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = torch.nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                x = batch[1].to(device)
                target = batch[4].to(device)

                pred = model(x)

                pred = pred[0]
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
