import urllib.request
import zipfile
import os

print("Downloading COCO val2017 annotations (approx. 241MB)... This may take a minute.")
url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
filepath = "annotations_trainval2017.zip"

if not os.path.exists("outputs/captions_val2017.json"):
    # Download the zip
    urllib.request.urlretrieve(url, filepath)
    print("Download complete. Extracting...")
    
    # Extract only the val2017 captions
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extract("annotations/captions_val2017.json", path=".")
    
    # Move the file to the outputs folder
    os.makedirs("outputs", exist_ok=True)
    os.rename("annotations/captions_val2017.json", "outputs/captions_val2017.json")
    
    # Clean up (optional)
    os.remove(filepath)
    os.rmdir("annotations")
    
    print("✅ Successfully saved to outputs/captions_val2017.json")
else:
    print("✅ File already exists in outputs/captions_val2017.json")