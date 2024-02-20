"""
Original: https://docs.ray.io/en/latest/ray-overview/getting-started.html 
"""

import argparse
from dvclive import Live
import json
import shutil
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import yaml

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint
from ray.train import get_context
import torch
from torch import nn

from src.helpers import get_dataloaders
from src.model import ConvNet


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


    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model the correct GPU/CPU device
    # ============================================================
    model = ConvNet()
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # [3] Set up Live object for DVCLive
    # ===============================
    live = None
    rank = ray.train.get_context().get_world_rank()
    if rank == 0:

        # Propogate DVC environment variables from Head Node to Workers
        DVC_ENV_VARS = config.get("dvc_env", None)
        if DVC_ENV_VARS:
            for name, value in  DVC_ENV_VARS.items():
                os.environ[name] = value
        
        # Initialize DVC Live
        live = Live(
            dir=os.path.join(DVC_ENV_VARS.get("DVC_ROOT", ""), "results/dvclive"),
            # save_dvc_exp=False
        )

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
        checkpoint_dir = "."
        checkpoint_path = checkpoint_dir + "/model.pth"
        torch.save(model.state_dict(), checkpoint_path)

        # Log metrics with print() 
        if live:
            print("-------------------")
            print(f"LOG METRICS WITH DVCLIVE")
            print("loss", test_loss)
            print("accuracy", accuracy)
            print("-------------------")
            
            live.log_metric("loss", test_loss)
            live.log_metric("accuracy", accuracy)
            live.next_step()
            


def train(params: dict) -> None:

    # [1] Define configuration for Training
    # =============================================
    train_params = params.get("train", {})
    NUM_WORKERS = train_params.get("num_workers", 1)
    USE_GPU = train_params.get("use_gpu", False)
    GLOBAL_BATCH_SIZE = train_params.get("global_GLOBAL_batch_size", 32)
    EPOCH_SIZE = train_params.get("epoch_size", 5) 
    TRAIN_RESULTS_DIR = train_params.get("results_dir")
    
    # [2] Load Best Model Parameters (from Tune stage)
    # =============================================
    TUNE_RESULTS_DIR = params.get("tune", {}).get("results_dir", "")    
    BEST_PARAMS_PATH = Path(TUNE_RESULTS_DIR) / "best_params.yaml"
    BEST_MODEL_PARAMS = yaml.safe_load(open(BEST_PARAMS_PATH))
    train_config = {
        "lr": BEST_MODEL_PARAMS.get("lr", 1e-2),
        "momentum": BEST_MODEL_PARAMS.get("momentum", 0.5),
        "epochs": EPOCH_SIZE,
        "batch_size_per_worker": GLOBAL_BATCH_SIZE // NUM_WORKERS,

        # Propogate DVC environment variables from Head Node to Workers 
        "dvc_env": {
            name: value for name, value in os.environ.items() if name.startswith("DVC")
        }
    }

    # [3] Configure computation resources
    # =============================================
    scaling_config = ScalingConfig(num_workers=NUM_WORKERS, use_gpu=USE_GPU)

    # [4] Initialize a Ray TorchTrainer
    # =============================================
    RAY_RESULTS_DIR = str(Path("results/ray_results").resolve())
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="MNIST",
            local_dir=RAY_RESULTS_DIR,
        )
    )

    # [5] Run Training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result: train.Result  = trainer.fit()

    print(f"Training result: {result}")
    print(f"result.path: {result.path}")
    print(f"result.metrics: {result.metrics}")
    print(f"result.metrics_dataframe: {result.metrics_dataframe}")
    print(f"result.config: {result.config}")
    
    # [6] Save Trial Results
    # =============================================
    print(f"Copying model.pth from {Path(result.path)} to {TRAIN_RESULTS_DIR}")
    shutil.copy(
        Path(result.path) / 'model.pth', 
        Path(TRAIN_RESULTS_DIR).resolve() /'model.pth'
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch MNIST Tune Example")
    parser.add_argument("--config", help="DVC parameters")
    args, _ = parser.parse_known_args()
    
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)

    train(params)
