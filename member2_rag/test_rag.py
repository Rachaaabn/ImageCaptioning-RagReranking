import torch
from PIL import Image
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel

print("1. Loading CLIP model...")
# We use a small, fast version of CLIP
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# --- Part 1: Convert Images to Vectors (Embeddings) ---
print("2. Extracting image embeddings...")
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"] # Your test images
image_ids = [101, 102, 103] # Fake IDs (like COCO IDs)

embeddings = []
for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # THE FIX IS HERE: Added .pooler_output
        image_features = model.get_image_features(**inputs).pooler_output
    
    # Convert to numpy and apply Normalization (Crucial for Cosine Similarity)
    vec = image_features.numpy().flatten()
    vec = vec / np.linalg.norm(vec) 
    embeddings.append(vec)

embeddings_matrix = np.array(embeddings).astype('float32')
print(f"Embeddings shape: {embeddings_matrix.shape}") # Should be (3, 512)

# --- Part 2: Build the FAISS Database ---
print("3. Building FAISS index...")
dimension = embeddings_matrix.shape[1] # 512 dimensions
# We use Inner Product (IP) because we normalized the vectors (equals Cosine Similarity)
index = faiss.IndexFlatIP(dimension) 
index.add(embeddings_matrix)
print(f"Total vectors in FAISS: {index.ntotal}")

# --- Part 3: Search Function (Retrieval) ---
print("4. Testing the search...")
query_text = "a dog playing in the park" # Test search query
text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)

with torch.no_grad():
    # THE FIX IS HERE TOO: Added .pooler_output
    text_features = model.get_text_features(**text_inputs).pooler_output

query_vec = text_features.numpy().flatten()
query_vec = query_vec / np.linalg.norm(query_vec)
query_vec = np.array([query_vec]).astype('float32')

# Ask FAISS for the top 2 closest images (k=2)
k = 2
distances, indices = index.search(query_vec, k)

print(f"\nQuery: '{query_text}'")
for i in range(k):
    idx = indices[0][i]
    score = distances[0][i]
    print(f"Result {i+1}: Image ID {image_ids[idx]} (Score: {score:.4f})")