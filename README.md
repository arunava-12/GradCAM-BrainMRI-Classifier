# Brain Tumor Detection

<p align="center">
  <img src="https://assets.technologynetworks.com/production/dynamic/images/content/354432/early-detection-of-brain-tumors-and-beyond-354432-960x540.jpg?cb=11900964" alt="Brain Tumor Detection" width="200">
</p>

<p align="center">
  Detect brain tumors from MRI images using deep learning with Grad-CAM visualization.
  <br>
  <strong>Accuracy: ~95%</strong>
</p>

## 🚀 Features

- **Dual Interfaces**: Web app (Flask) and Streamlit app for easy deployment
- **Deep Learning Model**: EfficientNet-based CNN for tumor classification
- **Grad-CAM Visualization**: Explainable AI to show model attention areas
- **Real-time Prediction**: Upload MRI images and get instant results
- **Class Support**: Glioma, Meningioma, Pituitary, No Tumor

## 📋 Table of Contents

- [About](#about)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## 📖 About

This project provides an automated brain tumor detection system using MRI images. It uses a pre-trained EfficientNet model fine-tuned for classifying brain tumors into four categories: Glioma, Meningioma, Pituitary, and No Tumor. The system includes Grad-CAM visualization to highlight regions of interest in the MRI scans.

Two interfaces are provided:
- **Flask Web App** (`app.py`): Traditional web interface with upload and result pages
- **Streamlit App** (`streamlit_app.py`): Modern, interactive web app with real-time visualization

## 🛠 Tech Stack

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **EfficientNet** - Base model architecture
- **Flask** - Web framework
- **Streamlit** - Interactive web app
- **OpenCV** - Image processing
- **NumPy** - Numerical computing
- **Pillow** - Image handling

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/arunava-12/Cancer_Detection.git
   cd Cancer_Detection
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install tensorflow flask streamlit opencv-python pillow numpy
   ```

### Running the Applications

#### Flask Web App
```bash
python app.py
```
Open http://127.0.0.1:5000/ in your browser

#### Streamlit App
```bash
streamlit run streamlit_app.py
```
The app will open automatically in your browser

## 📸 Usage

1. **Upload MRI Image**: Choose a PNG/JPG/JPEG MRI scan
2. **Get Prediction**: The model classifies the image into tumor types
3. **View Visualization**: Grad-CAM heatmap shows model attention areas
4. **Check Confidence**: Probability scores for all classes

### Supported Classes
- **Glioma**: Tumor originating in glial cells
- **Meningioma**: Tumor from meninges
- **Pituitary**: Tumor in pituitary gland
- **No Tumor**: Normal brain tissue

## 🧠 Model Details

- **Architecture**: EfficientNet-B0 fine-tuned
- **Input Size**: 224x224 pixels
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Accuracy**: ~95% on test set
- **Features**: Grad-CAM for explainability

The model file `best_model_finetuned.keras` contains the trained weights.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is for educational and research purposes only. Not intended for medical diagnosis.

## 👨‍💻 Author

**Arunava Mondal** - [GitHub](https://github.com/arunava-12)

---

**⚠️ Disclaimer**: This tool is for research purposes only and should not be used for actual medical diagnosis. Always consult with qualified medical professionals.
