import tensorflow as tf
from medmnist import PneumoniaMNIST
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.repeat(image, 3, axis=-1)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def main(args):
    # Load test dataset
    test_dataset = PneumoniaMNIST(split='test', download=True)
    test_images, test_labels = test_dataset.imgs, test_dataset.labels.flatten()
    
    # Create tf.data pipeline
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
    
    # Load model
    model = tf.keras.models.load_model(args.model_path)
    
    # Predictions
    y_pred_prob = model.predict(test_ds)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_labels
    
    # Metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))
    
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"AUC: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.show()
    
    # Failure cases: Identify misclassified images
    misclassified_indices = np.where(y_pred != y_true)[0]
    print(f"Number of misclassifications: {len(misclassified_indices)}")
    
    # Visualize a few
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, idx in enumerate(misclassified_indices[:6]):
        img = test_images[idx]
        axes[i//3, i%3].imshow(img, cmap='gray')
        axes[i//3, i%3].set_title(f"True: {['NORMAL', 'PNEUMONIA'][y_true[idx]]}, Pred: {['NORMAL', 'PNEUMONIA'][y_pred[idx]]}")
        axes[i//3, i%3].axis('off')
    plt.tight_layout()
    plt.savefig('failure_cases.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='final_model.h5')
    args = parser.parse_args()
    main(args)