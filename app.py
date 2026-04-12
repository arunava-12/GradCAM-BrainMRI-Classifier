import os

# ✅ MUST BE FIRST (before tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, request, redirect
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import logging
import cv2

# ✅ Suppress Flask logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Load model
model = tf.keras.models.load_model("best_model_finetuned.keras")

# ✅ Build model FIRST by calling it with dummy input
dummy_input = tf.zeros((1, 224, 224, 3))
model(dummy_input)

# ✅ Extract EfficientNet backbone AFTER calling model
base_model = None
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("EfficientNet base model not found.")

# ✅ Call base_model with dummy input to ensure it's built
base_model(dummy_input)

# ✅ Load class names
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# ✅ Find last Conv2D layer in base model
last_conv_layer_name = None
for layer in reversed(base_model.layers):
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name is None:
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                last_conv_layer_name = layer.name
                break
        except:
            continue

if last_conv_layer_name is None:
    raise ValueError("No Conv2D layer found in model for Grad-CAM.")

# ✅ Find base_model index inside sequential model
base_model_index = None
for i, layer in enumerate(model.layers):
    if layer.name == base_model.name:
        base_model_index = i
        break

print(f"✅ Using Grad-CAM layer: {last_conv_layer_name}")
print(f"✅ EfficientNet found at layer index: {base_model_index}")
print("✅ Grad-CAM ready.")


# ✅ Safe layer call — handles layers that don't accept training argument
def safe_call(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


# ✅ Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def make_gradcam_heatmap(img_array):
    # ✅ Create a model from base_model input → conv layer output
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # ✅ Create classifier model (rest of layers after base_model)
    x = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    y = x

    # pass through remaining layers in main model
    for layer in model.layers[base_model_index + 1:]:
        y = layer(y)

    classifier_model = tf.keras.models.Model(x, y)

    with tf.GradientTape() as tape:
        # Forward pass
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs)

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def generate_gradcam_image(image_path, save_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)

    input_array = np.expand_dims(img_array, axis=0)
    input_array = preprocess_input(input_array)

    heatmap = make_gradcam_heatmap(input_array)

    # ✅ Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (224, 224))

    # ✅ Full min-max normalization
    heatmap = np.float32(heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # ✅ Light blur
    heatmap_blurred = cv2.GaussianBlur(heatmap, (11, 11), 0)

    # ✅ Convert original to BGR
    img_bgr = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # ✅ Apply JET colormap
    heatmap_uint8 = np.uint8(255 * heatmap_blurred)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # ✅ Create alpha mask from heatmap intensity
    # Low activations = transparent (brain shows through)
    # High activations = opaque (heatmap shows)
    alpha = heatmap_blurred[..., np.newaxis]  # shape: (224, 224, 1)
    alpha = np.power(alpha, 0.6)              # gamma < 1: lift mid-range visibility
    alpha = np.clip(alpha, 0, 1)

    # ✅ Blend: where alpha=0 → pure brain, where alpha=1 → pure heatmap
    colored_float = colored.astype(np.float32)
    img_float = img_bgr.astype(np.float32)

    superimposed = img_float * (1 - alpha * 0.75) + colored_float * (alpha * 0.75)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    cv2.imwrite(save_path, superimposed)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# ✅ Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = secure_filename(f"{timestamp}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # ✅ Cleanup old uploads (keep latest 50)
            files = sorted(
                [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER'])],
                key=os.path.getctime
            )
            if len(files) > 50:
                for f in files[:10]:
                    os.remove(f)

            # ✅ Predict
            processed_img = preprocess_image(filepath)
            preds = model.predict(processed_img, verbose=0)[0]

            class_idx = np.argmax(preds)
            class_label = CLASS_NAMES[class_idx]
            confidence = float(preds[class_idx])

            probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

            # ✅ Grad-CAM
            gradcam_filename = "gradcam_" + filename
            gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
            generate_gradcam_image(filepath, gradcam_path)

            return render_template(
                'result.html',
                result=class_label,
                confidence=round(confidence * 100, 2),
                probabilities=probs,
                uploaded_image=filename,
                gradcam_image=gradcam_filename
            )

        except Exception as e:
            return f"Error: {str(e)}"

    return redirect(request.url)


if __name__ == '__main__':
    print("\n🚀 App running at: http://127.0.0.1:5000/\n")
    app.run(debug=False, host='127.0.0.1', port=5000)