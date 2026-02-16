import os
import zipfile
import requests


def download_dependencies(file_id):

    # Download URL
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Setup the paths
    zip_file_path = "dependencies/dependencies.zip"
    os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)
    
    # Download the file
    response = requests.get(download_url, stream=True)
    
    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("")
    
    # Clean up
    os.remove(zip_file_path)

    print("Dependencies downloaded and extracted successfully.")

if __name__ == "__main__":
    download_id = "1xJo9Lx_0vfDwsNhrykl0WurLwSMcgLqj"
    download_dependencies(download_id)