import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

def verify_model():
    model_path = "best_model_finetuned.keras"
    image_path = "data/Testing/glioma/Te-gl_1.jpg"
    class_names_path = "class_names.json"

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading class names from {class_names_path}...")
    with open(class_names_path, "r") as f:
        CLASS_NAMES = json.load(f)

    print(f"Loading and preprocessing image {image_path}...")
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    print("Running prediction...")
    preds = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(preds)
    class_label = CLASS_NAMES[class_idx]
    confidence = float(preds[class_idx])

    print("\nResults:")
    print(f"Predicted Class: {class_label}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll Probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {preds[i]:.2%}")

if __name__ == "__main__":
    verify_model()
