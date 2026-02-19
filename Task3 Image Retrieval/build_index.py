%Vector Index Construction Script
%This script loads the embeddings and builds a FAISS index for efficient similarity search.

import numpy as np
import faiss

def main():
    # Load embeddings
    embeddings = np.load('test_embeddings.npy').astype('float32')  # FAISS requires float32
    
    # Build FAISS index (L2 distance for similarity)
    dimension = embeddings.shape[1]  # 512
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Add all embeddings
    
    # Save index
    faiss.write_index(index, 'faiss_index.idx')
    print("FAISS index built and saved.")

if __name__ == "__main__":
    main()