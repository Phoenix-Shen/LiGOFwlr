import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Subset
import timm.data.transforms as timmtrans


# apply strong augmentation to dataset
transform_dict = {
    "MNIST_train": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "MNIST_test": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "CIFAR10_train": transforms.Compose(
        [
            timmtrans.RandomResizedCropAndInterpolation(
                (224, 224), interpolation="bicubic"
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandAugment(2),
            transforms.ToTensor(),
            transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
            transforms.RandomErasing(
                0.25,
            ),
        ]
    ),
    "CIFAR10_test": transforms.Compose(
        [
            transforms.Resize(224),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandAugment(2),
            transforms.ToTensor(),
            transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
            # transforms.RandomErasing(
            #     0.25,
            # ),
        ]
    ),
    "CIFAR100_train": transforms.Compose(
        [
            # transforms.Resize(224),
            timmtrans.RandomResizedCropAndInterpolation(
                (224, 224), interpolation="bicubic"
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandAugment(2),
            transforms.ToTensor(),
            transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
            transforms.RandomErasing(
                0.25,
            ),
        ]
    ),
    "CIFAR100_test": transforms.Compose(
        [
            transforms.Resize(224),
            # timmtrans.RandomResizedCropAndInterpolation(
            #     (224, 224), interpolation="bicubic"
            # ),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandAugment(2),
            transforms.ToTensor(),
            transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
            # transforms.RandomErasing(
            #     0.25,
            # ),
        ]
    ),
    "Flowers102_train": transforms.Compose(
        [
            timmtrans.RandomResizedCropAndInterpolation(
                (224, 224), interpolation="bicubic"
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandAugment(2),
            transforms.ToTensor(),
            transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
            transforms.RandomErasing(
                0.25,
            ),
        ]
    ),
    "Flowers102_test": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandAugment(2),
            transforms.ToTensor(),
            transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
            # transforms.RandomErasing(
            #     0.25,
            # ),
        ]
    ),
}


# def construct_dataset(
#     dataset_name: str,
#     root: str,
#     num_users: int,
#     iid_deg: float,
#     idx: int,
# ) -> tuple[Dataset, Dataset]:
#     """Generate datasets according to the given dataset name (support torchvision.Dataset)"""
#     # Get the transformations of the specified dataset
#     transform_train = transform_dict[dataset_name + "_train"]
#     transform_test = transform_dict[dataset_name + "_test"]
#     # Load the training and testing dataset
#     if dataset_name == "Flowers102":
#         train_set = getattr(tv.datasets, dataset_name)(
#             root, split="train", download=True, transform=transform_train
#         )
#         test_set = getattr(tv.datasets, dataset_name)(
#             root, split="test", download=True, transform=transform_test
#         )
#     else:
#         train_set = getattr(tv.datasets, dataset_name)(
#             root, train=True, download=True, transform=transform_train
#         )
#         test_set = getattr(tv.datasets, dataset_name)(
#             root, train=False, download=True, transform=transform_test
#         )

#     labels = train_set.targets if hasattr(train_set, "targets") else train_set._labels

#     # Generate a probability distribution for each client using Dirichlet distribution
#     probs = np.random.dirichlet([iid_deg] * len(set(labels)), num_users)

#     # Assign each sample to a client based on the probability distribution
#     assignments = []
#     for i in range(len(labels)):
#         label = labels[i]
#         p = probs[:, label]  # Get the probabilities for this label
#         # Choose a client based on the probabilities
#         c = np.random.choice(num_users, p=p / p.sum())
#         assignments.append(c)

#     # Create a list of datasets for each client based on the assignments
#     client_datasets = []
#     for i in range(num_users):
#         # Get the indices of samples assigned to this client
#         indices = [j for j in range(len(assignments)) if assignments[j] == i]
#         # Create a subset of the original dataset using these indices
#         subset = Subset(train_set, indices)
#         client_datasets.append(subset)


#     return client_datasets[idx], Subset(test_set, np.random.choice(len(test_set), 1000))
def construct_dataset(
    dataset_name: str,
    root: str,
    num_users: int,
    iid_deg: float,
    idx: int,
    test_size_per_client: int = 1000,
) -> tuple[Dataset, Dataset]:
    """Generate datasets according to the given dataset name (support torchvision.Dataset)"""
    # Get the transformations of the specified dataset
    transform_train = transform_dict[dataset_name + "_train"]
    transform_test = transform_dict[dataset_name + "_test"]

    # Load the training and testing dataset
    if dataset_name == "Flowers102":
        train_set = getattr(tv.datasets, dataset_name)(
            root, split="train", download=True, transform=transform_train
        )
        test_set = getattr(tv.datasets, dataset_name)(
            root, split="test", download=True, transform=transform_test
        )
    else:
        train_set = getattr(tv.datasets, dataset_name)(
            root, train=True, download=True, transform=transform_train
        )
        test_set = getattr(tv.datasets, dataset_name)(
            root, train=False, download=True, transform=transform_test
        )

    labels = train_set.targets if hasattr(train_set, "targets") else train_set._labels

    # Generate a probability distribution for each client using Dirichlet distribution
    probs = np.random.dirichlet([iid_deg] * len(set(labels)), num_users)

    # Assign each sample to a client based on the probability distribution
    assignments = []
    for i in range(len(labels)):
        label = labels[i]
        p = probs[:, label]  # Get the probabilities for this label
        c = np.random.choice(num_users, p=p / p.sum())  # Assign sample to a client
        assignments.append(c)

    # Create a list of datasets for each client based on the assignments
    client_datasets = []
    for i in range(num_users):
        # Get the indices of samples assigned to this client
        indices = [j for j in range(len(assignments)) if assignments[j] == i]
        # Create a subset of the original dataset using these indices
        subset = Subset(train_set, indices)
        client_datasets.append(subset)

    # Adjust the test set to match the training distribution for the client
    test_labels = test_set.targets if hasattr(test_set, "targets") else test_set._labels
    client_test_indices = []

    # Calculate the label proportions in the client's training data
    client_label_counts = np.bincount(
        [labels[i] for i in indices], minlength=len(set(labels))
    )
    client_label_proportions = client_label_counts / client_label_counts.sum()

    # Collect test samples to match client-specific label proportions
    for label, proportion in enumerate(client_label_proportions):
        # Get all indices in the test set with the current label
        label_indices = [
            i for i, test_label in enumerate(test_labels) if test_label == label
        ]
        # Sample a proportion of these indices for this client
        num_samples = int(proportion * test_size_per_client)
        sampled_indices = np.random.choice(label_indices, num_samples, replace=False)
        client_test_indices.extend(sampled_indices)

    # Create the client's test subset with the selected indices
    client_test_set = Subset(test_set, client_test_indices)

    return client_datasets[idx], client_test_set
