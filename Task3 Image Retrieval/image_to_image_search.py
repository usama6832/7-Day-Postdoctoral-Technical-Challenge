%Image-to-Image Search Script
%This script takes a query image (by index in the test set), retrieves the top-k similar images, and visualizes results.

import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medmnist import PneumoniaMNIST
import numpy as np
import faiss
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def preprocess_image(image):
    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    return img

def main(args):
    # Load model and index
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model.eval()
    index = faiss.read_index('faiss_index.idx')
    labels = np.load('test_labels.npy')
    
    # Load test dataset
    test_dataset = PneumoniaMNIST(split='test', download=True)
    
    # Query image (by index)
    query_idx = args.query_idx
    query_img = test_dataset.imgs[query_idx]
    query_label = labels[query_idx]
    
    # Extract query embedding
    processed_query = preprocess_image(query_img)
    with torch.no_grad():
        query_emb = model.encode_image([processed_query]).cpu().numpy().astype('float32').flatten()
    
    # Search
    k = args.k
    distances, indices = index.search(query_emb.reshape(1, -1), k + 1)  # +1 to exclude self
    top_indices = indices[0][1:]  # Exclude the query itself
    
    # Visualize
    fig, axes = plt.subplots(1, k + 1, figsize=(15, 5))
    axes[0].imshow(query_img, cmap='gray')
    axes[0].set_title(f"Query: {['Normal', 'Pneumonia'][query_label]}")
    axes[0].axis('off')
    
    for i, idx in enumerate(top_indices):
        img = test_dataset.imgs[idx]
        label = labels[idx]
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].set_title(f"Result {i+1}: {['Normal', 'Pneumonia'][label]}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'retrieval_results_query_{query_idx}.png')
    plt.show()
    
    # Print results
    print(f"Query Label: {['Normal', 'Pneumonia'][query_label]}")
    print(f"Top-{k} Results: {[labels[idx] for idx in top_indices]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_idx', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    main(args)