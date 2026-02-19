Task 1: CNN Classification with Comprehensive Analysis using the ResNet-15 architecture and the PneumoniaMNIST dataset from MedMNIST v2.

Training Script (train.py)
This script loads PneumoniaMNIST, preprocesses it, and trains the ResNet-15 model.

Usage: Run python train.py --batch_size 32 --epochs 10. The dataset downloads automatically.

Evaluation Script (evaluate.py)
This script loads the test set from PneumoniaMNIST, evaluates the ResNet-15 model, and generates metrics/visualizations.

Usage: Run python evaluate.py --model_path final_model.h5.





Task 2: Medical Report Generation using Visual Language Model.

Usage: Run pip install transformers torch pillow (for Hugging Face and dependencies). 
Then, python generate_reports.py --prompt_type domain_specific. 
Ensure images are saved as PNGs from PneumoniaMNIST (e.g., via test_dataset.imgs[idx] saved as images). 
The script outputs reports to generated_reports.txt.




Task 3: Semantic Image Retrieval System using the PneumoniaMNIST dataset (from MedMNIST v2) for consistency with previous tasks.

Usage: Run python extract_embeddings.py. This generates test_embeddings.npy and test_labels.npy.

Vector Index Construction Script (build_index.py)
This script loads the embeddings and builds a FAISS index for efficient similarity search.

Usage: Run python build_index.py after extracting embeddings. This creates faiss_index.idx.

Image-to-Image Search Script (image_to_image_search.py)
This script takes a query image (by index in the test set), retrieves the top-k similar images, and visualizes results.

Usage: Run python image_to_image_search.py --query_idx 0 --k 5. This visualizes and prints results for query index 0.



Text-to-Image Search Script (text_to_image_search.py)
This script takes a text query, retrieves the top-k similar images, and visualizes results.

Usage: Run python text_to_image_search.py --text_query "chest X-ray with lung infiltrates" --k 5.
