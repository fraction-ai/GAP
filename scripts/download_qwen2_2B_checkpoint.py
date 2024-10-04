import boto3
import os

def download_s3_folder(bucket_name, s3_folder, local_dir):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Iterate over all objects in the specified S3 folder
    for obj in bucket.objects.filter(Prefix=s3_folder):
        # Remove the folder prefix from the key and join with local directory
        target_path = os.path.join(local_dir, obj.key[len(s3_folder):])
        
        # Skip directories (just make sure you're working with files)
        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path))
        
        print(f"Downloading {obj.key} to {target_path}")
        bucket.download_file(obj.key, target_path)

# Replace these values with your S3 bucket name and the folder to download
bucket_name = 'finetune-checkpoints'
s3_folder = 'checkpoints/QWEN2-VL/qwen2_2B_cmp_adv/checkpoint-2100/'  # Folder to download
local_dir = '../evaluate/checkpoints/qwen2_2B/'  # Local directory where the folder will be saved

download_s3_folder(bucket_name, s3_folder, local_dir)
