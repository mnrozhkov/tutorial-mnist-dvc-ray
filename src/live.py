import boto3
import os
from pathlib import Path
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
        try:
            # Attempt to create an S3 client with default credential resolution
            s3_client = boto3.client('s3')
            # Test the credentials by listing buckets (or any light operation)
            s3_client.list_buckets()
            return s3_client
        except NoCredentialsError:
            raise NoCredentialsError("AWS credentials not found. Please set up AWS credentials.")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
    
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
        print("##################################################")
        print(f"\nFile Uploaded {file_path}... \n")
        print("##################################################")

    def _upload_directory(self, directory_path: str):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, directory_path)
                s3_path = self.s3_directory + relative_path.replace("\\", "/")  # Replace backslash with forward slash for S3
                self.s3.upload_file(local_file_path, self.bucket_name, s3_path)

    def pull(self, filename: str):

        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.s3_directory)

        for page in pages:
            for obj in page.get('Contents', []):
                s3_file_key = obj['Key']
                local_file_path = os.path.join(os.path.relpath(s3_file_key, self.s3_directory), filename)
                print(local_file_path)

                # Ensure the directory exists
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                self.s3.download_file(self.bucket_name, s3_file_key, local_file_path)
                print(f"File downloaded from S3 to {local_file_path}")

class DVCLiveRayLogger(Live):
    def __init__(self, bucket_name, s3_directory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.s3_directory = s3_directory
        self.storage = self._get_storage()

    def _get_storage(self):
        # Create a new StorageObject instance when needed
        return StorageObject(self.bucket_name, self.s3_directory)

    def next_step(self, *args, **kwargs):
        
        print("\nDVCLiveLogger: PUSH METRICS")
        # storage = self._get_storage()
        self.storage.push(self.dir, force=True)

        super().next_step(*args, **kwargs)


def download_folder_from_s3(bucket_name, s3_folder, local_dir_path):
    
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    # Ensure the folder_prefix ends with a slash
    if not s3_folder.endswith('/'):
        s3_folder += '/'

    print(f"Listing objects in folder '{s3_folder}' of bucket '{bucket_name}':")
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get('Contents', []):

            key = os.path.relpath(obj['Key'], s3_folder)
            
            if key != '.':
            
                # Define the local path to save the file
                local_file_path = os.path.join(local_dir_path, key)
    
                # Ensure that the directory for this file exists (this creates any necessary subdirectories)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download the file from S3
                s3.download_file(bucket_name, obj['Key'], local_file_path)
                print(f"Downloaded {obj['Key']} to {local_file_path}")


def download_file_from_s3(bucket_name, s3_file_key, local_file_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, s3_file_key, local_file_path)
    print(f"Downloaded {s3_file_key} to {local_file_path}")


def list_objects_in_s3_folder(bucket_name, folder_prefix):
    """
    List all objects in a specific folder of an S3 bucket.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - folder_prefix (str): The folder prefix within the S3 bucket.
    """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')

    # Ensure the folder_prefix ends with a slash
    if not folder_prefix.endswith('/'):
        folder_prefix += '/'

    print(f"Listing objects in folder '{folder_prefix}' of bucket '{bucket_name}':")

    # Use the paginator to handle buckets with many objects
    operation_parameters = {'Bucket': bucket_name, 'Prefix': folder_prefix}
    page_iterator = paginator.paginate(**operation_parameters)

    obj_keys = []
    for page in page_iterator:
        for obj in page.get('Contents', []):
            
            key = os.path.relpath(obj['Key'], folder_prefix)
            if key != '.':
                obj_keys.append(key)
    
    return obj_keys
