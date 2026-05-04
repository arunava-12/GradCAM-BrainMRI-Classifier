import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

try:
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def evaluate_efficiently():
    model_path = "best_model_finetuned.keras"
    test_dir = "data/Testing"
    class_names_path = "class_names.json"

    print("Loading class names...")
    with open(class_names_path, "r") as f:
        CLASS_NAMES = json.load(f)

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    print(f"Loading test dataset from {test_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )

    # Ensure class names match
    ds_class_names = test_ds.class_names
    print(f"Dataset classes: {ds_class_names}")

    # Preprocess
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

    print("Running predictions (this will also measure throughput)...")
    start_time = time.time()
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    end_time = time.time()
    
    total_time = end_time - start_time
    total_images = len(y_true)
    
    print("-" * 50)
    print("EVALUATION RESULTS & METRICS")
    print("-" * 50)
    print(f"Total Test Images: {total_images}")
    
    if total_images > 0:
        avg_time_per_img = total_time / total_images
        print(f"Total Inference Time: {total_time:.2f} seconds")
        print(f"Average Inference Time per Image: {avg_time_per_img:.4f} seconds")
        print(f"Throughput: {1.0 / avg_time_per_img:.2f} images/second (Batch size {BATCH_SIZE})")
        
        # Map indices back to CLASS_NAMES if the dataset loaded them in a different order
        # image_dataset_from_directory sorts classes alphanumerically.
        # CLASS_NAMES from json: ["glioma", "meningioma", "notumor", "pituitary"]
        # So we should use ds_class_names for the report to be accurate.
        
        if HAS_SKLEARN:
            acc = accuracy_score(y_true, y_pred)
            print(f"\nOverall Accuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=ds_class_names))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
        else:
            correct = sum([1 for i, j in zip(y_true, y_pred) if i == j])
            print(f"\nOverall Accuracy: {correct / total_images:.4f} ({correct}/{total_images})")
            print("Note: Install scikit-learn for detailed precision/recall/f1 metrics.")

if __name__ == "__main__":
    evaluate_efficiently()
