%Text-to-Image Search Script (text_to_image_search.py)
%This script takes a text query, retrieves the top-k similar images, and visualizes results.

import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPTextModel
from medmnist import PneumoniaMNIST
import numpy as np
import faiss
import matplotlib.pyplot as plt
import argparse

def main(args):
    # Load model and index
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, text_cls=MedCLIPTextModel)
    model.from_pretrained()
    model.eval()
    index = faiss.read_index('faiss_index.idx')
    labels = np.load('test_labels.npy')
    
    # Load test dataset
    test_dataset = PneumoniaMNIST(split='test', download=True)
    
    # Text query
    text_query = args.text_query
    
    # Extract text embedding
    with torch.no_grad():
        text_emb = model.encode_text([text_query]).cpu().numpy().astype('float32').flatten()
    
    # Search
    k = args.k
    distances, indices = index.search(text_emb.reshape(1, -1), k)
    top_indices = indices[0]
    
    # Visualize
    fig, axes = plt.subplots(1, k, figsize=(15, 5))
    for i, idx in enumerate(top_indices):
        img = test_dataset.imgs[idx]
        label = labels[idx]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Result {i+1}: {['Normal', 'Pneumonia'][label]}")
        axes[i].axis('off')
    
    plt.suptitle(f"Text Query: '{text_query}'")
    plt.tight_layout()
    plt.savefig(f'text_retrieval_results.png')
    plt.show()
    
    # Print results
    print(f"Text Query: {text_query}")
    print(f"Top-{k} Results: {[labels[idx] for idx in top_indices]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_query', type=str, default="chest X-ray showing pneumonia")
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    main(args)