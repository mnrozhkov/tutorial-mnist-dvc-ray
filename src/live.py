import asyncio
import boto3
import os
from pathlib import Path
from dvclive.lightning import DVCLiveLogger as _DVCLiveLogger
from dvclive.lightning import ModelCheckpoint, rank_zero_only
from botocore.exceptions import NoCredentialsError
from pyparsing import C
from ray.tune import Callback
import time
import threading
from dvclive import Live

class StorageObject:
    def __init__(self, bucket_name: str, s3_directory: str):
        self.bucket_name = bucket_name
        self.s3_directory = s3_directory.rstrip("/") + "/"  # Ensure the directory ends with a '/'
        self._s3 = None

    @property
    def s3(self):
        if self._s3 is None:
            self._s3 = self._create_s3_client()
        return self._s3

    def _create_s3_client(self):
        if 'AWS_PROFILE' in os.environ:
            # Use the AWS_PROFILE environment variable
            boto3.setup_default_session(profile_name=os.environ['AWS_PROFILE'])
        elif 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ:
            # Use AWS Access Key ID and Secret Access Key from environment variables
            return boto3.client(
                's3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
            )
        else:
            raise NoCredentialsError("AWS credentials not found. Please set up the AWS_PROFILE environment variable or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

        return boto3.client('s3')
    
    def push(self, local_path: str, force: bool = False):
        if os.path.isdir(local_path):
            self._upload_directory(local_path)
        elif os.path.isfile(local_path):
            self._upload_file(local_path)
        else:
            raise ValueError(f"The path {local_path} is not a valid file or directory")

    def _upload_file(self, file_path: str):
        s3_path = self.s3_directory + os.path.basename(file_path)
        self.s3.upload_file(file_path, self.bucket_name, s3_path)

    def _upload_directory(self, directory_path: str):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                local_file_path = os.path.join(root, file)

                print("##################################################")
                print(f"\nCALLBACK - Uploading {local_file_path}... \n")
                print(f"CALLBACK - Uploading {os.path.abspath(local_file_path)}... \n")
                print("##################################################")

                relative_path = os.path.relpath(local_file_path, directory_path)
                s3_path = self.s3_directory + relative_path.replace("\\", "/")  # Replace backslash with forward slash for S3
                self.s3.upload_file(local_file_path, self.bucket_name, s3_path)

    def pull(self, local_path: str):
        # Implementation for downloading objects from S3 to the local file system
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.s3_directory):
            for obj in page.get('Contents', []):
                # Construct the local file path
                local_file_path = os.path.join(local_path, os.path.relpath(obj['Key'], self.s3_directory))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                self.s3.download_file(self.bucket_name, obj['Key'], local_file_path)

class DVCLiveRayLogger(Live):
    def __init__(self, bucket_name, s3_directory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.s3_directory = s3_directory
        # self.trial_dir = trial_dir

    def _get_storage(self):
        # Create a new StorageObject instance when needed
        return StorageObject(self.bucket_name, self.s3_directory)

    def next_step(self, *args, **kwargs):
        super().next_step(*args, **kwargs)
        print("\nDVCLiveLogger: PUSH METRICS")
        storage = self._get_storage()
        storage.push(self.dir, force=True)


class DVCLiveLoggerCallback(Callback):
    def __init__(self, dir: str, bucket_name: str, s3_directory: str, **kwargs):
        super().__init__(**kwargs)
        self.dir = dir
        self.bucket_name = bucket_name
        self.s3_directory = s3_directory
        self.counter = 0

    def _get_storage(self):
        # Create a new StorageObject instance when needed
        return StorageObject(self.bucket_name, self.s3_directory)

    def on_checkpoint(self, **info) -> None:
        # Push the directory where checkpoints are saved
        print("\nDVCLiveLoggerCallback: PUSH METRICS (on_checkpoint)")
        print(info)
        storage = self._get_storage()
        storage.push(self.dir, force=True)

        # DEV: control number of calls
        print(f"COUNTER: {self.counter}\n")
        self.counter += 1


class S3SyncRunner:
    def __init__(self, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.thread = None
        self.running = False

    def start(self):
        if self.running:
            print("Runner is already running.")
            return
        self.running = True
        self.thread = threading.Thread(target=self.run_periodically, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def run_periodically(self):
        while self.running:
            self.function(*self.args, **self.kwargs)
            time.sleep(self.interval)

class DVCLiveS3SyncRunner(S3SyncRunner):
    def __init__(self, storage, local_path, interval=3):
        super().__init__(interval, self.pull_from_storage, storage, local_path)

    def pull_from_storage(self, storage, local_path):
        print("Pulling from storage...")
        storage.pull(local_path)
        print("Pull complete.")
