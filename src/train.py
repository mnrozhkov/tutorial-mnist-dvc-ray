"""
Source: https://docs.ray.io/en/latest/ray-overview/getting-started.html 
"""

import os
from filelock import FileLock
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize
from tqdm import tqdm

from ray import train
import ray.train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer


def get_dataloaders(batch_size):
    # Transform to normalize the input images
    transform = transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    with FileLock(os.path.expanduser("~/data.lock")):
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root="~/data",
            train=True,
            download=True,
            transform=transform,
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="~/data",
            train=False,
            download=True,
            transform=transform,
        )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


# Model Definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    # Get dataloaders inside worker training function
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)

    # [1] Prepare Dataloader for distributed training
    # Shard the datasets among workers and move batches to the correct device
    # =======================================================================
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    model = NeuralNetwork()

    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model the correct GPU/CPU device
    # ============================================================
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Model training loop
    from dvclive import Live
    from ray.train import get_context

    train_context = get_context()
    rank = train_context.get_local_rank()


    with Live(dir='results/dvclive', dvcyaml=True, save_dvc_exp=False, resume=True) as live:

        for epoch in range(epochs):
            model.train()
            for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            test_loss, num_correct, num_total = 0, 0, 0
            with torch.no_grad():
                for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                    pred = model(X)
                    loss = loss_fn(pred, y)

                    test_loss += loss.item()
                    num_total += y.shape[0]
                    num_correct += (pred.argmax(1) == y).sum().item()

            test_loss /= len(test_dataloader)
            accuracy = num_correct / num_total

            # [3] Report metrics to Ray Train
            # ===============================
            ray.train.report(metrics={"loss": test_loss, "accuracy": accuracy})

            # Log metrics with DVCLive 
            if rank == 0:
                live.log_metric("loss", test_loss)
                live.log_metric("accuracy", accuracy)
                live.next_step()

            


def train_fashion_mnist(num_workers=2, use_gpu=False):
    global_batch_size = 32

    train_config = {
        "lr": 1e-3,
        "epochs": 5,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    # sync_config = ray.train.SyncConfig(sync_artifacts=True)

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="experiment_name",
            storage_path="~/dev/rnd/ray-dvc/results",
            # sync_config=sync_config
        )
    )

    # [4] Start Distributed Training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result: train.Result  = trainer.fit()
    print(f"Training result: {result}")
    print(f"result.filesystem: {result.filesystem}")
    print(f"result.path: {result.path}")
    print(f"result.metrics: {result.metrics}")
    print(f"result.metrics_dataframe: {result.metrics_dataframe}")
    print(f"result.config: {result.config}")
    
    if result.checkpoint:
        print(f"result.checkpoint.path: {result.checkpoint.path}")


if __name__ == "__main__":
    
    import os
    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = "results"

    train_fashion_mnist(num_workers=2, use_gpu=False)