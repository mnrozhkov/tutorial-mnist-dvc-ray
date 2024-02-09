# Source: https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html 

import os
import tempfile

from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer

from src.test_scripts.test_s3 import show_aws_credentials_content

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
    # os.environ["AWS_ACCESS_KEY_ID"]="ASIAU7UXIWDIX45GAJ7D"
    # os.environ["AWS_SECRET_ACCESS_KEY"]="FdpglatzMK8K0A8wyrlHNmGRw+a6abtIBaSoHqou"
    # os.environ["AWS_SESSION_TOKEN"]="IQoJb3JpZ2luX2VjECEaCXVzLWVhc3QtMSJGMEQCICyCpo+1nY7DyxJESwT79RIezMVOKXOdAuNfa4ufacJlAiBRuubxFUegYX77MTSjQfxGQl062egwd3fYnloGjsWBeyqoAgjp//////////8BEAIaDDM0Mjg0MDg4MTM2MSIMP8w+7GS5oo5OYPa0KvwBhYULM+Nxl01kiBv5rLjXF94XFduU/iONbgTEVifskv14WIcRYmfipqXwROF3yBGJiNMZ4NDiiOghFuNQBP9ZMT8BuTAtNa0v621HfUd3Obu/j1uLGMNyCIevR8E6Y4hNczs95znkHwNnjfzcT5Pssx0K4TLDcQQJo7eIliTaNERVI+l1rKQEk6IIv3U6vZq3NSHaihVRif9iZGJSPinXd9Mfo4jhw1J1bmm1dgFHywAOxbKOCJe2bH3HOHowrJjS8xhco9Koq7HlPCygfxuYJTAieuW4nwY5qjC/o5siQbZ8kwoNsCBg3mt8/qiqnDpr+FLWWlrSeIPz5Z67MPuckq4GOp4BbINjhNFHDaYIbUl7+FHDzkE0wfG498Edpza86DESc+L+GNUs2/BEiOWlQU+QqZLPgvrZ4rldAuomxlKdBT5Tkhe/q+RXKFE8JLoEMhgBdJ0z8n61xBzzcA0IoMrLYKxo++1wt+daeFdJSQMSAjEYgb1F7+iKBKEQ6qV3e7QuSaEOfnxHV5g30a5IGTvlLeCADGcS8w9mi4g4K2zKOKs="

    trainer = TorchTrainer(
        train_fn,
        scaling_config=train.ScalingConfig(num_workers=2),
        run_config=train.RunConfig(
            name="test_s3_storage",
            storage_path="s3://cse-cloud-version/tutorial-mnist-dvc-ray/ray_shared_storage/",
            sync_config=train.SyncConfig(sync_artifacts=True),
        )
    )
    result: train.Result = trainer.fit()
    
    print("TEST_S3_STORAGE - result: ")
    print(result)



