"""
Source: https://docs.ray.io/en/latest/ray-overview/getting-started.html 
"""

import argparse
import json
import shutil
import os
import logging
from pathlib import Path
from typing import Dict
from regex import P
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
    worker_rank = ray.train.get_context().get_world_rank()

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
    print("#############################################")
    print("DVC_ENV_VARS")
    print(config.get("dvc_env", None))
    print("#############################################")
    dvc_env = config.get("dvc_env", None)
    if dvc_env:
        for name, value in  dvc_env.items():
            os.environ[name] = value
    

    live = None
    if worker_rank == 0:
                
        from dvclive import Live
        live = Live(
            dir=os.path.join(os.environ["DVC_ROOT"], "results/dvclive"),
            dvcyaml=False, 
        )

        print("#############################################")
        print("RAY_TRAIN_CONTEXT")
        print(ray.train.get_context().get_experiment_name())
        print(ray.train.get_context().get_metadata())
        print(ray.train.get_context().get_storage())
        print("TRIAL_DIR", ray.train.get_context().get_trial_dir())
        print("TRIAL_NAME", ray.train.get_context().get_trial_name())
        print("Live.dir", live.dir)
        print("Live.params_file", live.params_file)
        print("#############################################")

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

        # [4] Report metrics to Ray Train
        # ===============================
        checkpoint_dir = "."
        checkpoint_path = checkpoint_dir + "/model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        ray.train.report(
            metrics={"loss": test_loss, "accuracy": accuracy},
            checkpoint=Checkpoint.from_directory(checkpoint_dir)
        )

        # [5] Log metrics with DVCLive 
        # ===============================
        if worker_rank == 0:
            if live:
                print("-------------------")
                print(f"LOG METRICS WITH DVCLIVE")
                print("worker local rank - ", ray.train.get_context().get_local_rank())
                print("worker global rank - ", ray.train.get_context().get_world_rank())
                print("node rank - ", ray.train.get_context().get_node_rank())
                print("epoch - ", epoch)
                print("loss - ", test_loss)
                print("accuracy - ", accuracy)
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

    TUNE_RESULTS_DIR = params.get("tune", {}).get("results_dir", "")
    BEST_PARAMS_PATH = Path(TUNE_RESULTS_DIR) / "best_params.yaml"
    BEST_MODEL_PARAMS = yaml.safe_load(open(BEST_PARAMS_PATH))

    train_config = {
        "lr": BEST_MODEL_PARAMS.get("lr", 1e-2),
        "momentum": BEST_MODEL_PARAMS.get("momentum", 0.5),
        "epochs": EPOCH_SIZE,
        "batch_size_per_worker": GLOBAL_BATCH_SIZE // NUM_WORKERS,
        "dvc_env": {name: value for name, value in os.environ.items() if
                    name.startswith("DVC") and name != "DVC_STUDIO_TOKEN"}
    }

    # [2] Configure computation resources
    # =============================================
    scaling_config = ScalingConfig(num_workers=NUM_WORKERS, use_gpu=USE_GPU)

    # [3] Initialize a Ray TorchTrainer
    # =============================================
    RAY_RESULTS_DIR = str(Path("results/ray_results").resolve())
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="MNIST",
            local_dir=RAY_RESULTS_DIR,
            log_to_file=True,
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
    
    # Save Trial Results
    train_metrics_df = result.metrics_dataframe
    TRAIN_METRICS_PATH = "results/train/train_metrics.csv"
    train_metrics_df.to_csv(TRAIN_METRICS_PATH)
    train_report = {
        'config': result.config,
        'path': result.path,
        'metrics': result.metrics,
    }
    with open('report.json', 'w') as f:
        json.dump(train_report, f)
    
    DVCLIVE_PATH_SOURCE = Path(train_report.get('path'))
    print(f"DVCLIVE_PATH_SOURCE: {DVCLIVE_PATH_SOURCE}")
    for filename in ['result.json', 'model.pth']: 
        shutil.copyfile(
            DVCLIVE_PATH_SOURCE / filename, 
            Path(TRAIN_RESULTS_DIR).resolve() / filename
        )
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch MNIST Tune Example")
    parser.add_argument("--config", help="DVC parameters")
    args, _ = parser.parse_known_args()
    
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)

    train(params)
