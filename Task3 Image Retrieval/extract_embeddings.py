%This script loads MedCLIP, processes all test images from PneumoniaMNIST, extracts embeddings, and saves them as a NumPy array.


import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medmnist import PneumoniaMNIST
from PIL import Image
import numpy as np
import os

def preprocess_image(image):
    # Convert to PIL Image and resize
    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    return img

def main():
    # Load MedCLIP model
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()  # Loads pre-trained weights
    model.eval()
    
    # Load test dataset
    test_dataset = PneumoniaMNIST(split='test', download=True)
    images = test_dataset.imgs  # Shape: (624, 28, 28)
    labels = test_dataset.labels.flatten()  # 0: Normal, 1: Pneumonia
    
    embeddings = []
    for img in images:
        processed_img = preprocess_image(img)
        # Extract image embedding (MedCLIP outputs 512-dim vector)
        with torch.no_grad():
            emb = model.encode_image([processed_img])  # Returns tensor
            embeddings.append(emb.cpu().numpy().flatten())
    
    embeddings = np.array(embeddings)  # Shape: (624, 512)
    
    # Save embeddings and labels
    np.save('test_embeddings.npy', embeddings)
    np.save('test_labels.npy', labels)
    print("Embeddings and labels saved.")

if __name__ == "__main__":
    main()