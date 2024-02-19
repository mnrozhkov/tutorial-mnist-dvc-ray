import boto3
from datetime import datetime
import os
import ray

@ray.remote
def list_buckets():
    # Initialize S3 resource
    s3 = boto3.resource('s3')
    
    # Print out bucket names
    for bucket in s3.buckets.all():
        print(f'Bucket Name: {bucket.name}')

# def list_objects_in_bucket(bucket_name, prefix=''):
#     """
#     List objects in an S3 bucket, optionally filtered by a prefix.

#     Parameters:
#     - bucket_name (str): Name of the S3 bucket.
#     - prefix (str, optional): Prefix to filter objects in the bucket.

#     Returns:
#     - list: A list of object keys in the specified bucket.
#     """
#     # Initialize S3 client
#     s3_client = boto3.client('s3')
    
#     # Initialize the list to store object keys
#     object_keys = []

#     try:
#         # Use paginator to handle buckets with many objects
#         paginator = s3_client.get_paginator('list_objects_v2')
#         page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

#         for page in page_iterator:
#             if "Contents" in page:
#                 for obj in page['Contents']:
#                     object_keys.append(obj['Key'])

#         print(f"Objects in '{bucket_name}'" + (f" with prefix '{prefix}'" if prefix else "") + ":")
#         for key in object_keys:
#             print(key)

#     except Exception as e:
#         print(f"Error listing objects in S3 bucket: {e}")

#     return object_keys

@ray.remote
def upload_file(bucket_name, file_name, file_content):

    print("Showing AWS credentials content... on remote")
    show_aws_credentials_content()

    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    try:
        # Writing the string content to S3
        s3_client.put_object(
            Bucket=bucket_name, 
            Key=file_name, 
            Body=file_content)
        return f'Successfully uploaded {file_name} to {bucket_name}.'
    except Exception as e:
        return f'Error uploading file to S3: {e}'

def show_aws_credentials_content():
    # Define the path to the AWS credentials file and AWS directory
    aws_credentials_path = os.path.expanduser('~/.aws/credentials')
    aws_dir_path = os.path.expanduser('~/.aws')
    
    # Display the content of ~/.aws/credentials
    try:
        print("Contents of ~/.aws/credentials:")
        with open(aws_credentials_path, 'r') as file:
            print(file.read())
    except FileNotFoundError:
        print("The AWS credentials file does not exist.")
    except Exception as e:
        print(f"Error reading the AWS credentials file: {e}")

    # List the contents of ~/.aws directory
    try:
        print("\nContents of ~/.aws directory:")
        for item in os.listdir(aws_dir_path):
            print(item)
    except FileNotFoundError:
        print("The ~/.aws directory does not exist.")
    except Exception as e:
        print(f"Error listing the contents of the ~/.aws directory: {e}")


if __name__ == "__main__":


    # Configuration
    bucket_name = 'cse-cloud-version'  # Replace with your S3 bucket name
    file_content = 'This is a test file'
    file_name = f'test-file-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt'

    
    # print(os.environ)
    show_aws_credentials_content()
    
    # list_buckets.remote()

    # print(f"Objects in '{bucket_name}':")
    # list_objects_in_bucket(bucket_name)


    print("Uploading file to S3...")
    object_ref = upload_file.remote(bucket_name, file_name, file_content)
    result = ray.get(object_ref)
    print(result)
    


