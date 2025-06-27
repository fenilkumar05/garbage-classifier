# 🗂️ Smart Garbage Classification System

An AI-powered waste classification application built with deep learning and Streamlit, designed to promote sustainable waste management practices through intelligent garbage categorization.

## 📋 Description

The Smart Garbage Classification System is a comprehensive machine learning application that uses Convolutional Neural Networks (CNN) to automatically classify different types of waste materials. The system provides real-time image classification, voice announcements, performance analytics, and detailed reporting capabilities to support environmental sustainability initiatives.

## ✨ Features

- **🤖 Advanced AI Classification**: CNN-based deep learning model for accurate waste categorization
- **🔊 Multilingual Voice Support**: Text-to-speech announcements with language options
- **📊 Real-time Training Visualization**: Live progress tracking during model training
- **📈 Comprehensive Analytics**: Detailed performance metrics and model evaluation
- **📱 Live Camera Detection**: Real-time webcam integration for instant classification
- **📋 Prediction History**: Complete logging and tracking of all classifications
- **📄 Automated Reporting**: PDF report generation with charts and analytics
- **♻️ Eco-friendly Tips**: Recycling guidance for each waste category
- **🎨 Modern UI/UX**: Professional, responsive interface with loading indicators
- **☁️ Cloud-ready Deployment**: Optimized for cloud platforms and local environments

## 🗂️ Supported Waste Categories

- **Cardboard** - Packaging materials, boxes
- **Glass** - Bottles, containers, jars
- **Metal** - Cans, aluminum, steel items
- **Paper** - Documents, newspapers, magazines
- **Plastic** - Bottles, containers, packaging
- **Trash** - General non-recyclable waste

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart-garbage-classification
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv garbage_classifier_env
   garbage_classifier_env\Scripts\activate
   
   # macOS/Linux
   python3 -m venv garbage_classifier_env
   source garbage_classifier_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   streamlit --version
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

## 🎯 Usage

### Running the Application

1. **Start the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will automatically open in your default browser

### Basic Workflow

1. **Model Training** (First-time setup)
   - Navigate to "🎯 Train Model" in the sidebar
   - Upload a ZIP file containing your dataset
   - Configure training parameters (epochs, batch size, etc.)
   - Click "🚀 Start Training" and monitor progress

2. **Image Classification**
   - Go to "🔍 Predict Image"
   - Upload an image file (PNG, JPG, JPEG)
   - Click "🔍 Classify Image"
   - View results with confidence scores and recycling tips

3. **Performance Analysis**
   - Check "📈 Model Performance" for training metrics
   - Review "📋 Prediction History" for classification logs
   - Generate "📄 PDF Reports" for comprehensive analysis

### Dataset Structure

Organize your training data as follows:
```
dataset/
├── cardboard/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## 📦 Dependencies

### Core Libraries
- **streamlit** (≥1.28.0) - Web application framework
- **tensorflow** (≥2.13.0) - Deep learning and neural networks
- **numpy** (≥1.24.0) - Numerical computing
- **pandas** (≥2.0.0) - Data manipulation and analysis

### Visualization & UI
- **matplotlib** (≥3.7.0) - Static plotting
- **seaborn** (≥0.12.0) - Statistical visualization
- **plotly** (≥5.15.0) - Interactive charts
- **Pillow** (≥10.0.0) - Image processing

### Additional Features
- **opencv-python-headless** (≥4.8.0) - Computer vision
- **scikit-learn** (≥1.3.0) - Machine learning utilities
- **pyttsx3** (≥2.90) - Text-to-speech functionality
- **reportlab** (≥4.0.0) - PDF generation

## 🛠️ Configuration

### Audio Settings
- Toggle voice output on/off
- Select language for announcements
- Automatic TTS after classification

### Training Parameters
- **Epochs**: 5-50 (default: 20)
- **Batch Size**: 16, 32, 64 (default: 32)
- **Validation Split**: 0.1-0.4 (default: 0.2)
- **Learning Rate**: 0.001, 0.0001, 0.00001 (default: 0.001)

## 📊 Model Architecture

The system uses a Convolutional Neural Network with the following structure:
- Input Layer: 224x224x3 RGB images
- 4 Convolutional blocks with MaxPooling
- Dropout layer (0.5) for regularization
- Dense layers: 512 neurons → 6 output classes
- Activation: ReLU (hidden), Softmax (output)
- Optimizer: Adam with configurable learning rate

## 🔧 Troubleshooting

### Common Issues

1. **TTS not working**: Normal in cloud environments; voice features disabled automatically
2. **Model loading errors**: Ensure model file exists and TensorFlow version compatibility
3. **Memory issues**: Reduce batch size or image resolution for training
4. **Import errors**: Verify all dependencies are installed correctly

### Performance Tips

- Use GPU acceleration for faster training (CUDA-compatible)
- Ensure balanced dataset with 100+ images per class
- Monitor validation accuracy to prevent overfitting
- Use data augmentation for better generalization

## 📈 Future Enhancements

- Real-time camera integration
- Mobile application development
- Advanced model architectures (ResNet, EfficientNet)
- Multi-language UI support
- Cloud storage integration
- Batch processing capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Credits

**Developed by Fenil**

*Smart Garbage Classification System - Leveraging AI for Environmental Sustainability*

---

*Built with ❤️ for a cleaner, more sustainable future*
