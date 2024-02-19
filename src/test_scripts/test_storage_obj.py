import os
from datetime import datetime

from src.live import StorageObject
from src.test_scripts.test_s3 import show_aws_credentials_content

def save_text_file(file_path, file_content):
    
    # Write the file content
    try:
        with open(file_path, 'w') as file:
            file.write(file_content)
        print(f"File successfully saved to {file_path}")
    except Exception as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":

    file_content = 'This is a test file'
    file_name = f'test-file-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt'
    directory = '/tmp/test_s3'

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Define the full path for the file
    file_path = os.path.join(directory, file_name)
    print(file_path)
    save_text_file(file_path, file_content)

    show_aws_credentials_content()

    bucket_name = "cse-cloud-version"
    s3_directory = "tutorial-mnist-dvc-ray/ray_shared_storage"
    # os.environ['AWS_PROFILE'] = "iterative-sandbox"

    storage = StorageObject(bucket_name, s3_directory)
    storage.push(file_path, force=True)
