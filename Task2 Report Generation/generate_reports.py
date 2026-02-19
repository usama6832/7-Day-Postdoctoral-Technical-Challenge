#This script loads MedGemma, processes a list of images (from PneumoniaMNIST), and generates reports using configurable prompts.

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
import argparse
import os

def load_medgemma():
    # Load MedGemma model and processor
    model_id = "google/med-gemma-2b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

def preprocess_image(image_path, target_size=(224, 224)):
    # Load and resize image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    return img

def generate_report(model, processor, image, prompt, max_length=200):
    # Prepare inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # Generate report
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    
    # Decode output
    report = processor.decode(outputs[0], skip_special_tokens=True)
    return report

def main(args):
    model, processor = load_medgemma()
    
    # Sample images (replace with your paths; here using dummy paths for PneumoniaMNIST samples)
    image_paths = [
        "sample_images/normal_1.png", "sample_images/normal_2.png", "sample_images/normal_3.png", "sample_images/normal_4.png", "sample_images/normal_5.png",
        "sample_images/pneumonia_1.png", "sample_images/pneumonia_2.png", "sample_images/pneumonia_3.png", "sample_images/pneumonia_4.png", "sample_images/pneumonia_5.png"
    ]  # In practice, extract from PneumoniaMNIST test set
    
    prompts = {
        "simple": "Describe the findings in this chest X-ray image.",
        "detailed": "Provide a detailed medical report for this chest X-ray, including observations on lung fields, heart, and any abnormalities.",
        "domain_specific": "As a radiologist, analyze this chest X-ray for signs of pneumonia or normal findings. Describe key features, potential diagnoses, and recommendations."
    }
    
    selected_prompt = prompts[args.prompt_type]
    
    reports = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = preprocess_image(img_path)
            report = generate_report(model, processor, img, selected_prompt)
            reports.append((img_path, report))
            print(f"Report for {img_path}: {report}")
        else:
            print(f"Image {img_path} not found.")
    
    # Save reports to file
    with open("generated_reports.txt", "w") as f:
        for img_path, report in reports:
            f.write(f"Image: {img_path}\nReport: {report}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', type=str, default='domain_specific', choices=['simple', 'detailed', 'domain_specific'])
    args = parser.parse_args()
    main(args)