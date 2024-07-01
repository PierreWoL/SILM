import torch

from Deepjoin.train import construct_train_dataset

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "all-mpnet-base-v2"  # multi-qa-distilbert-cos-v1
    data_path = "datasets/WDC/Test/"
    naming_file = "datasets/WDC/naming.csv"
    train_dataset = construct_train_dataset(data_path, naming_file, model_name=model_name,
                                            column_to_text_transformation="title-colname-stat-col",
                                            shuffle_rate=0.2, seed=42, device=device)
    """ train_model(model_name=model_name, train_dataset=train_dataset, dev_samples=None, model_save_path=None,
                batch_size=64,
                learning_rate=2e-5, warmup_steps=None, weight_decay=0.01, num_epochs=32,
                device=device, cpuid=2)"""