# 🦷 Dental Pathology Classification System

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
</div>

<div align="center">
  <h3>🌐 <a href="https://huggingface.co/spaces/Mr0Diablo/Dental-Pathology-Classifier">🚀 Try the Live Application</a></h3>
  <p><em>Experience the dental pathology classification system in action!</em></p>
</div>

---

## 🎯 Overview

A dental pathology classification system using deep learning to identify oral conditions from images. The project evolved from custom CNN architectures to EfficientNetB0 transfer learning, deployed through a Streamlit web interface for easy use.

### Key Features
- **7-class dental image classification** using EfficientNetB0 transfer learning
- **Web interface** built with Streamlit for easy image upload and prediction
- **Medical information** included for each condition with risk levels
- **Prediction confidence** and top-3 results display

---

## 🌐 Live Demo

### **🔗 [Access the Application](https://huggingface.co/spaces/Mr0Diablo/Dental-Pathology-Classifier)**

**Features Available:**
- 📤 Upload dental images (JPG, PNG, BMP, TIFF)
- 🔍 Real-time pathology classification
- 📊 Confidence scores and detailed results
- 🏥 Medical information for each condition
- 📝 Session history tracking

**System Status:** ✅ **Online and Ready**

---

## 📊 Dataset & Performance

The dataset contains **5,624 high-resolution dental images** across 7 pathological categories:

| Category | Full Name | Description | Risk Level |
|----------|-----------|-------------|------------|
| **CaS** | Cold Sore (Herpes Simplex) | Viral infection causing fluid-filled blisters | Low |
| **CoS** | Canker Sore (Aphthous Ulcer) | Shallow, painful ulcers inside mouth | Low |
| **Gum** | Gum Disease (Periodontal Disease) | Infection and inflammation of gums | Medium |
| **MC** | Mucocele (Mucous Cyst) | Benign cyst containing mucus | Low |
| **OC** | Oral Cancer | Malignant tumor in oral cavity | High |
| **OLP** | Oral Lichen Planus | Chronic inflammatory condition | Medium |
| **OT** | Other/Normal Tissue | Healthy tissue or benign conditions | Low |

**Data Split:**
- Training: 3,087 images (55%)
- Validation: 1,028 images (18%)
- Testing: 1,509 images (27%)

---

## 🏗️ Model Architecture

### Development Phases

**Phase 1: Custom CNN**
- Base CNN: 51.07% accuracy
- With data augmentation: 80.45% accuracy
- 16.87M parameters

**Phase 2: Transfer Learning**
- EfficientNetB0 pre-trained on ImageNet
- Fine-tuned for dental pathology classification
- Input size: 256×256 pixels
- Improved performance with fewer parameters

---

## 🚀 Web Application

The Streamlit app provides:
- **Image upload** for classification
- **Prediction results** with confidence scores
- **Medical information** for each condition
- **Risk assessment** (Low/Medium/High)
- **Session history** tracking
- **Responsive design** with dark mode support

### 🌐 Access Methods

**🔗 Live Application:** [http://34.203.68.42:8501](http://34.203.68.42:8501)

**📱 Mobile Friendly:** The application is fully responsive and works on mobile devices

**🌍 External Access:** Available globally - share the link with colleagues and collaborators

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install streamlit tensorflow pillow numpy opencv-python
```

### Model Requirements
The application requires the trained model file:
```
efficientnetb0_transfer_final.keras
```

### Running the Application Locally
```bash
streamlit run dental_classification_app.py
```

**Local URLs:**
- Network URL: http://10.108.57.171:8501
- External URL: http://34.203.68.42:8501

---

## 💻 Technical Implementation

### Image Processing Pipeline
1. **Format Validation**: Supports JPG, PNG, BMP, TIFF formats
2. **Preprocessing**: RGB conversion, resizing to 256×256
3. **Normalization**: EfficientNet-specific preprocessing
4. **Batch Processing**: Optimized for single and batch predictions

### Model Integration
- **Cached Loading**: Efficient model loading with @st.cache_resource
- **Progress Tracking**: Real-time loading and prediction progress
- **Error Handling**: Comprehensive error management and user feedback
- **Memory Optimization**: Efficient resource usage for web deployment

### User Experience Features
- **📤 Drag & Drop Upload**: Intuitive file upload interface
- **🖼️ Image Preview**: Original image display with metadata
- **📊 Interactive Results**: Expandable sections and detailed breakdowns
- **📝 Prediction History**: Session-based prediction tracking
- **🔄 Real-time Updates**: Dynamic content updates without page refresh

---

## 📈 Results

| Model | Accuracy | Loss | Parameters |
|-------|----------|------|------------|
| Custom CNN | 80.45% | 0.5498 | 16.87M |
| EfficientNetB0 Transfer | 95.2% | 0.187 | 6.2M |

---

## 🔬 Technical Highlights

- **Transfer Learning**: EfficientNetB0 pre-trained on ImageNet
- **Medical Image Processing**: Specialized preprocessing for dental images
- **Web Deployment**: Streamlit application with modern UI
- **Data Augmentation**: Improved model generalization
- **Multi-class Classification**: 7 distinct oral pathology categories

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Deep Learning**: Custom CNN development and transfer learning
- **Computer Vision**: Medical image classification and preprocessing
- **Web Development**: Streamlit application deployment
- **Data Science**: Model evaluation and performance analysis

---

## 🚀 Future Enhancements

- **Model Improvements**: Ensemble methods and additional architectures
- **More Categories**: Expand to additional oral pathology types
- **Mobile App**: Native mobile application development
- **API Development**: RESTful API for integration
- **Batch Processing**: Multiple image analysis capabilities

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Improve documentation
- Submit pull requests

---

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare providers for medical evaluation.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Cellula Technologies** for the internship opportunity
- Healthcare professionals for dataset validation
- Open-source community for tools and libraries
- TensorFlow and Streamlit teams for the frameworks

---

<div align="center">
  <strong>Developed during Computer Vision Engineering Internship at Cellula Technologies</strong>
  <br><br>
  <a href="http://34.203.68.42:8501">
    <img src="https://img.shields.io/badge/🌐_Live_Demo-Available-brightgreen?style=for-the-badge" alt="Live Demo">
  </a>
</div>
