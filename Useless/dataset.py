import torch
def generate_positive_negative_samples(tabular_dataset):
    positive_pairs = []
    negative_pairs = []
    for i in range(len(tabular_dataset)):
        positive_idx = torch.randint(len(tabular_dataset), size=(1,))
        negative_idx = torch.randint(len(tabular_dataset), size=(1,))
        while positive_idx == i:
            positive_idx = torch.randint(len(tabular_dataset), size=(1,))
        while negative_idx == i or negative_idx == positive_idx:
            negative_idx = torch.randint(len(tabular_dataset), size=(1,))

        positive_pairs.append((i, positive_idx))
        negative_pairs.append((i, negative_idx))

    return positive_pairs, negative_pairs