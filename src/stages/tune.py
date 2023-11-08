# Original Code here: https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch.py

import os
from pathlib import Path
from dvclive import Live
import argparse
from filelock import FileLock
import tempfile
from typing import Any, Dict

import ray
from ray import train, tune
from ray.train import Checkpoint, Result
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import yaml

from src.model import ConvNet
import yaml


# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


def train_func(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test_func(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def get_data_loaders(batch_size=64):
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=False, download=True, transform=mnist_transforms
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    return train_loader, test_loader


def train_mnist(config):
    should_checkpoint = config.get("should_checkpoint", False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders()
    model = ConvNet().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    while True:
        train_func(model, optimizer, train_loader, device)
        acc = test_func(model, test_loader, device)
        metrics = {"mean_accuracy": acc}

        # Report metrics (and possibly a checkpoint)
        if should_checkpoint:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(model.state_dict(), os.path.join(tempdir, "model.pt"))
                train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics)


def tune_hyperparameters(params: dict, is_cuda: bool, is_smoke_test: bool):

    # [1] Define configuration for Training
    # =============================================
    tune_params = params.get("tune", {})

    if tune_params.get("run_tune", False) is False:
        print("Tune - Skipping Hyperparameter Tuning")
        return

    EPOCH_SIZE = tune_params.get("epoch_size", 256) 
    TEST_SIZE = tune_params.get("test_size", 128)
    TUNE_RESULTS_DIR = params.get("tune", {}).get("results_dir")
    

    ray.init(num_cpus=2 if is_smoke_test else None)

    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {"cpu": 2, "gpu": int(is_cuda)}  # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(train_mnist, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=1 if args.smoke_test else 50,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={
                "mean_accuracy": 0.98,
                "training_iteration": 5 if args.smoke_test else 100,
            },
        ),
        param_space={
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        },
    )
    results:ResultGrid = tuner.fit()


    best_result: Result = results.get_best_result()
    best_params: Dict[str, Any] = best_result.config
    print("Tune - Best config is:", best_params)
    BEST_PARAMS_PATH = Path(TUNE_RESULTS_DIR) / "best_params.yaml"
    with open(BEST_PARAMS_PATH, 'w') as f:
        yaml.dump(best_params, f)

    PLOT_PATH = "results/tune/mean_accuracy_plot.png"
    ax = None
    for result in results:
        label = f"lr={result.config['lr']:.3f}, momentum={result.config['momentum']}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "mean_accuracy", label=label)
        else:
            result.metrics_dataframe.plot("training_iteration", "mean_accuracy", ax=ax, label=label, 
                                          figsize=(10,6))
    ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
    ax.set_ylabel("Mean Test Accuracy")
    fig = ax.get_figure()
    fig.savefig(PLOT_PATH)

    # Save Trial Results
    TRIAL_RESULTS_PATH = "results/tune/trial_results.csv"
    trial_results_df = results.get_dataframe()
    trial_results_df.to_csv(TRIAL_RESULTS_PATH)

    with Live(dir=TUNE_RESULTS_DIR, save_dvc_exp=False) as live:    
        live.log_artifact(BEST_PARAMS_PATH, cache=False)
        live.log_artifact(TRIAL_RESULTS_PATH, cache=False)
        live.log_image("mean_accuracy_plot.png", PLOT_PATH)
        
    assert not results.errors

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch MNIST Tune Example")
    parser.add_argument("--config", help="DVC parameters")
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    tune_hyperparameters(params, args.cuda, args.smoke_test)