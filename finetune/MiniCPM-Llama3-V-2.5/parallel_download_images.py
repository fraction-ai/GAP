import os
import json
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set the directory to save images
image_dir = "train_images"
os.makedirs(image_dir, exist_ok=True)

# Load the JSON data
with open('./base_data_gap_vqa.json', 'r') as file:
    data = json.load(file)

# Function to generate a unique filename from the URL
def url_to_filename(url):
    parsed_url = urlparse(url)
    # Extract domain, path, and filename
    domain = parsed_url.netloc.replace('.', '_')
    path = parsed_url.path.replace('/', '_')
    # Combine them to form the filename
    filename = f"{domain}{path}"
    return filename

# Function to download an image and return the local path
def download_image(url, img_id):
    img_name = url_to_filename(url)
    img_path = os.path.join(image_dir, img_name)

    # Check if the image already exists
    if not os.path.exists(img_path):
        try:
            img_data = requests.get(url, timeout=10).content
            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)
            print(f"Downloaded: {img_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")
    else:
        print(f"Already exists: {img_path}")

    return img_path

# Function to download images in parallel using a thread pool
def download_images_in_parallel(data, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the thread pool
        futures = {executor.submit(download_image, item['image'], item['id']): item for item in data}
        
        for future in as_completed(futures):
            try:
                future.result()  # Check if the task raised an exception
            except Exception as exc:
                item = futures[future]
                print(f"Image {item['image']} generated an exception: {exc}")

# Update the JSON data with local image paths in parallel
download_images_in_parallel(data, max_workers=10)
