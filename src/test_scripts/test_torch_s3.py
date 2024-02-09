# Source: https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html 

import os
import tempfile

from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer

from src.test_scripts.test_s3 import show_aws_credentials_content
from src.live import download_folder_from_s3

def train_fn(config):

    print("Show AWS creds - inside train_fg")
    show_aws_credentials_content()

    for i in range(5):
    
        print(f"iteration {i}")

        # Save arbitrary artifacts to the working directory
        rank = train.get_context().get_world_rank()
        with open(f"artifact-rank={rank}-iter={i}.txt", "w") as f:
            f.write("data")

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint_path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            with open(checkpoint_path, "w") as f:
                f.write("data")
                

if __name__ == "__main__":

    import os
    print(os.environ)
    
    print("Show AWS creds - before training")
    show_aws_credentials_content()

    trainer = TorchTrainer(
        train_fn,
        scaling_config=train.ScalingConfig(num_workers=2),
        run_config=train.RunConfig(
            name="test_s3_storage",
            storage_path="s3://cse-cloud-version/tutorial-mnist-dvc-ray/ray_shared_storage/",
            sync_config=train.SyncConfig(sync_artifacts=False),
        )
    )
    result: train.Result = trainer.fit()
    
    print("TEST_S3_STORAGE - result: ")
    print(result)


    print("Download trial results from S3")
    s3_path_parts = result.path.split("/", 1)
    bucket_name = s3_path_parts[0]
    results_s3_directory = s3_path_parts[1]
    download_folder_from_s3(bucket_name, results_s3_directory, 'results/test_s3_storage')





