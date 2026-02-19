import tensorflow as tf
from tensorflow.keras import layers, models
from medmnist import PneumoniaMNIST
import numpy as np
import argparse

def residual_block(x, filters, kernel_size=3, stride=1):
    # Basic residual block for ResNet-15
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet15(input_shape=(224, 224, 3)):
    # Custom ResNet-15 architecture (15 layers: 1 initial conv + 4 blocks with 2 convs each + pooling/dense)
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks (4 blocks, each with 2 conv layers, totaling ~15 layers)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, x)
    return model

def preprocess_image(image, label):
    # Resize to 224x224 for ResNet-15
    image = tf.image.resize(image, [224, 224])
    # Convert grayscale to RGB by repeating channels
    image = tf.repeat(image, 3, axis=-1)
    # Normalize to [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment_image(image, label):
    # Augmentation for robustness
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_rotation(image, 0.1)
    image = tf.image.random_zoom(image, (0.9, 1.1))
    return image, label

def main(args):
    # Load datasets
    train_dataset = PneumoniaMNIST(split='train', download=True)
    val_dataset = PneumoniaMNIST(split='val', download=True)
    
    # Convert to tf.data
    train_images, train_labels = train_dataset.imgs, train_dataset.labels.flatten()
    val_images, val_labels = val_dataset.imgs, val_dataset.labels.flatten()
    
    # Create tf.data pipelines
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(preprocess_image).map(augment_image).shuffle(1000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.map(preprocess_image).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build ResNet-15 model
    model = build_resnet15()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    
    # Training (no separate fine-tuning for custom ResNet; train all layers)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('final_model.h5')
    print("Model saved as final_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)