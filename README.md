# 🦷 Teeth Classification With CNN

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter">
</div>

---

## 🎯 Overview

A comprehensive computer vision solution for classifying dental images into 7 distinct categories, developed as part of a Computer Vision Engineering internship at **Cellula Technologies**. This project demonstrates the application of deep learning techniques to medical imaging, contributing to AI-driven diagnostic tools in healthcare.

### Key Features
- **7-class dental image classification** using custom CNN architectures
- **End-to-end ML pipeline** from preprocessing to model evaluation
- **5,624 high-resolution images** with strategic train/validation/test splits
- **Dual CNN architecture** with and without data augmentation

---

## 📊 Dataset

The dataset contains **5,624 high-resolution dental images** across 7 categories:

| Category | Description | Use Case |
|----------|-------------|----------|
| **OT** | Oral Tissue | General oral health assessment |
| **CoS** | Crown of Sinus | Sinus-related dental conditions |
| **MC** | Molar Crown | Molar-specific diagnostics |
| **CaS** | Cavity in Sinus | Cavity detection in sinus area |
| **OC** | Oral Cavity | General oral cavity examination |
| **OLP** | Oral Lichen Planus | Inflammatory condition detection |
| **Gum** | Gum Tissue | Gum health evaluation |

**Data Split:**
- Training: 3,087 images (55%)
- Validation: 1,028 images (18%)
- Testing: 1,509 images (27%)

---

## 🏗️ Architecture

### Model Design
Two CNN architectures built from scratch using TensorFlow:

1. **Base CNN Model**: Standard convolutional architecture
2. **Augmented CNN Model**: Incorporates data augmentation layers

### Architecture Details
The model uses a progressive feature extraction approach:
- **Layer 1**: 32 filters for basic feature detection
- **Layer 2**: 64 filters for intermediate features  
- **Layer 3**: 128 filters for complex pattern recognition
- **Final Layer**: Dense classification layer with dropout regularization

**Model Parameters:** ~16.87 million trainable parameters

---

## 🛠️ Installation

The project requires standard machine learning and computer vision libraries including TensorFlow, NumPy, Pandas, Matplotlib, OpenCV, and Jupyter notebooks for development and analysis.

---

## 🚀 Usage

The project includes comprehensive preprocessing pipelines, data visualization tools, and model training workflows. The implementation follows standard machine learning practices with separate modules for data handling, model development, and evaluation.

### Project Structure
The codebase is organized into logical modules:
- **Models**: CNN architectures and training logic
- **Preprocessing**: Data cleaning and augmentation utilities  
- **Visualization**: Analysis and plotting tools
- **Notebooks**: Interactive development and experimentation
- **Results**: Model outputs and performance metrics

---

## 📈 Results

| Model | Accuracy | Loss |
|-------|----------|------|
| Base CNN | 51.07% | 1.3420
| Augmented CNN | 80.45% | 0.5498

*Results will be updated as training progresses*

---

## 🔬 Technical Highlights

- **Custom CNN Architecture**: Built from scratch using TensorFlow
- **Data Augmentation**: Rotation, zoom, flip, and brightness adjustments
- **Regularization**: Dropout layers to prevent overfitting
- **Multi-class Classification**: Softmax activation for 7-class output
- **Medical Imaging Focus**: Specialized preprocessing for dental images

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Medical Image Processing**: Handling healthcare-specific datasets
- **Deep Learning**: Custom CNN design and implementation
- **Model Optimization**: Hyperparameter tuning and regularization
- **Data Visualization**: Comprehensive exploratory data analysis
- **MLOps Practices**: Structured project organization and documentation

---

## 🤝 Contributing

This project is part of ongoing development during my internship. Contributions and suggestions are welcome through:
- Issue reports
- Feature requests
- Code improvements
- Documentation enhancements

---

## 📄 License

This project is **proprietary and confidential**. All rights reserved by Cellula Technologies.

---

## 🙏 Acknowledgments

- **Cellula Technologies** for providing the internship opportunity
- Healthcare professionals who contributed to dataset creation
- Open-source community for the foundational tools and libraries

---

<div align="center">
  <strong>🔬 Developed during Computer Vision Engineering Internship at Cellula Technologies</strong>
</div>
