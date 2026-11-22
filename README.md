# Nebula: AI-Powered Astronomy Image Classifier

An interactive deep learning application built with **PyTorch** and **Streamlit** that classifies astronomical objects (galaxies, nebulae, stars, and supernovas) from uploaded images.

---

## Features

* **Transfer Learning**: Uses a pre-trained `ResNet18` architecture fine-tuned for astronomical classification.
* **Dynamic Categories**: Automatically adapts to whatever training folders you provide using PyTorch's `ImageFolder`.
* **Interactive Web UI**: A clean, single-page Streamlit interface complete with real-time confidence scores and progress bars.
* **Local Caching**: Optimized model loading so inference runs smoothly without lag.

---

## Project Structure

```text
Nebula/
├── dataset/
│   ├── train/          # Training images sorted by category folders
│   └── val/            # Validation images sorted by category folders
├── venv/               # Python virtual environment
├── app.py              # Streamlit web application UI
├── train.py            # Model training script
├── astronomy_model.pth # Generated after training your model
└── README.md           # Project documentation
