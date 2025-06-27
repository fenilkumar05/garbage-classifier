"""
Smart Garbage Classification System

An AI-powered waste classification application built with deep learning and Streamlit.
This system uses Convolutional Neural Networks (CNN) to automatically classify different
types of waste materials, promoting sustainable waste management practices through
intelligent garbage categorization.

Features:
- Advanced CNN-based garbage classification
- Multilingual text-to-speech announcements
- Real-time training progress visualization
- Comprehensive model performance analysis
- Live camera detection capability
- Prediction history and logging
- Automated PDF report generation
- Eco-friendly recycling tips

Author: Fenil
Version: 1.0.0
License: MIT
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
import json
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
import pyttsx3
import threading
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import zipfile

# Configure page
st.set_page_config(
    page_title="🗂️ Smart Garbage Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .prediction-card h3, .prediction-card h4, .prediction-card strong {
        color: #000000 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #45a049 0%, #4CAF50 100%);
    }

    /* Professional Loading Indicators */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e3e3e3;
        border-top: 4px solid #4CAF50;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 1rem;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.1rem;
    }

    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #4CAF50;
        margin: 0 2px;
        animation: pulse 1.4s infinite ease-in-out both;
    }

    .pulse-dot:nth-child(1) { animation-delay: -0.32s; }
    .pulse-dot:nth-child(2) { animation-delay: -0.16s; }

    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }

    .progress-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

class GarbageClassifier:
    def __init__(self):
        self.model = None
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.model_path = 'garbage_classifier_model.h5'
        self.history_path = 'training_history.pkl'
        self.predictions_log = 'predictions_log.json'
        self.mute = False
        self.current_language = 'English'

        # Language-specific voice configurations
        self.language_config = {
            'English': {
                'voice_id': None,  # Use default English voice
                'rate': 150,
                'volume': 0.9,
                'waste_phrase': "This is {class_name} waste",
                'confidence_phrase': "with {confidence} percent confidence"
            },
            'Hindi': {
                'voice_id': None,  # Will be set based on available voices
                'rate': 140,
                'volume': 0.9,
                'waste_phrase': "Yah {class_name} kachra hai",
                'confidence_phrase': "{confidence} pratishat vishwas ke saath"
            },
            'Gujarati': {
                'voice_id': None,  # Will be set based on available voices
                'rate': 140,
                'volume': 0.9,
                'waste_phrase': "Aa {class_name} kacharo chhe",
                'confidence_phrase': "{confidence} taka bharosa sathe"
            }
        }

        # Initialize TTS after language config is set
        self.tts_engine = self.init_tts()

    def init_tts(self):
        """Initialize text-to-speech engine with voice detection"""
        try:
            # Check if running in cloud environment
            if os.getenv('STREAMLIT_SHARING') or os.getenv('STREAMLIT_CLOUD'):
                print("Running in cloud environment - TTS disabled")
                return None

            # Try to initialize pyttsx3 engine
            engine = pyttsx3.init()

            # Test if engine is working by trying to get properties
            try:
                engine.getProperty('rate')
                engine.getProperty('volume')
            except Exception as test_e:
                print(f"TTS engine test failed: {test_e}")
                return None

            # Get available voices
            voices = engine.getProperty('voices')
            if voices:
                # Configure voices for different languages
                for voice in voices:
                    voice_name = voice.name.lower()
                    voice_id = voice.id

                    # Try to identify language-specific voices
                    if 'hindi' in voice_name or 'devanagari' in voice_name:
                        self.language_config['Hindi']['voice_id'] = voice_id
                    elif 'gujarati' in voice_name:
                        self.language_config['Gujarati']['voice_id'] = voice_id
                    elif 'english' in voice_name or (hasattr(voice, 'languages') and voice.languages and 'en' in voice.languages[0]):
                        if not self.language_config['English']['voice_id']:
                            self.language_config['English']['voice_id'] = voice_id

            # Set default properties with error handling
            try:
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
            except Exception as prop_e:
                print(f"Warning: Could not set TTS properties: {prop_e}")

            print("TTS engine initialized successfully")
            return engine

        except ImportError:
            print("pyttsx3 not available - TTS disabled")
            return None
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            return None

    def set_language(self, language):
        """Set the current language for TTS"""
        if language in self.language_config:
            self.current_language = language
            if self.tts_engine:
                config = self.language_config[language]

                # Set voice if available
                if config['voice_id']:
                    try:
                        self.tts_engine.setProperty('voice', config['voice_id'])
                    except:
                        pass  # Use default voice if setting fails

                # Set rate and volume
                self.tts_engine.setProperty('rate', config['rate'])
                self.tts_engine.setProperty('volume', config['volume'])

    def _ensure_tts_engine(self):
        """Ensure TTS engine is properly initialized and working"""
        try:
            if self.tts_engine is None:
                self.tts_engine = self.init_tts()
                return self.tts_engine is not None

            # Test if engine is still working
            try:
                # Try to get a property to test if engine is alive
                self.tts_engine.getProperty('rate')
                return True
            except:
                # Engine is dead, reinitialize
                print("TTS engine appears to be dead, reinitializing...")
                self.tts_engine = self.init_tts()
                return self.tts_engine is not None

        except Exception as e:
            print(f"TTS engine check failed: {e}")
            return False

    def speak(self, text, language=None):
        """Enhanced text-to-speech functionality with language support"""
        if self.mute:
            return

        # Ensure TTS engine is working
        if not self._ensure_tts_engine():
            print("TTS engine not available")
            return

        # Use provided language or current language
        target_language = language if language else self.current_language

        try:
            # Set language configuration
            if target_language != self.current_language:
                self.set_language(target_language)

            def speak_thread():
                try:
                    # Double-check engine is still available in thread
                    if self.tts_engine:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS speak failed: {e}")
                    # Mark engine as dead so it gets reinitialized next time
                    self.tts_engine = None

            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
        except Exception as e:
            print(f"TTS error: {e}")
            # Mark engine as dead so it gets reinitialized next time
            self.tts_engine = None

    def speak_prediction(self, predicted_class, confidence, language=None):
        """Speak prediction result in the specified language"""
        target_language = language if language else self.current_language

        if target_language not in self.language_config:
            target_language = 'English'

        config = self.language_config[target_language]

        # Format the prediction message
        waste_text = config['waste_phrase'].format(class_name=predicted_class)
        confidence_text = config['confidence_phrase'].format(confidence=int(confidence * 100))

        full_message = f"{waste_text} {confidence_text}"

        self.speak(full_message, target_language)

    def get_available_languages(self):
        """Get list of available languages"""
        return list(self.language_config.keys())

    def add_language_support(self, language_name, config):
        """Add support for a new language (for future extensibility)"""
        required_keys = ['voice_id', 'rate', 'volume', 'waste_phrase', 'confidence_phrase']

        if all(key in config for key in required_keys):
            self.language_config[language_name] = config
            return True
        else:
            print(f"Invalid language config for {language_name}. Missing required keys.")
            return False

    def get_voice_info(self):
        """Get information about available voices for debugging"""
        if not self.tts_engine:
            return "TTS engine not initialized"

        try:
            voices = self.tts_engine.getProperty('voices')
            voice_info = []

            for voice in voices:
                info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages if hasattr(voice, 'languages') else 'Unknown'
                }
                voice_info.append(info)

            return voice_info
        except Exception as e:
            return f"Error getting voice info: {e}"
    
    def load_model(self):
        """Load pre-trained model if exists"""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                return True
            except:
                return False
        return False
    
    def create_model(self, input_shape=(224, 224, 3), num_classes=6):
        """Create CNN model architecture"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image, source="unknown"):
        """Preprocess image for prediction with enhanced consistency"""
        try:
            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image)
            elif hasattr(image, 'read'):  # File-like object (from camera or upload)
                image = Image.open(image)

            # Ensure we have a PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError(f"Invalid image type: {type(image)}")

            # Store original dimensions for debugging
            original_size = image.size

            # Convert to RGB if needed (crucial for consistency)
            if image.mode != 'RGB':
                print(f"Converting image from {image.mode} to RGB (source: {source})")
                image = image.convert('RGB')

            # Apply consistent resizing with high-quality resampling
            image = image.resize((224, 224), Image.Resampling.LANCZOS)

            # Convert to array and normalize consistently
            img_array = np.array(image, dtype=np.float32) / 255.0

            # Ensure proper shape
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                raise ValueError(f"Invalid image shape after preprocessing: {img_array.shape}")

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Debug information
            print(f"Image preprocessed - Source: {source}, Original: {original_size}, "
                  f"Final shape: {img_array.shape}, Range: [{img_array.min():.3f}, {img_array.max():.3f}]")

            return img_array

        except Exception as e:
            print(f"Error preprocessing image from {source}: {e}")
            raise
    
    def predict_image(self, image, source="unknown"):
        """Predict garbage class for an image with enhanced accuracy"""
        if self.model is None:
            print("Error: Model not loaded")
            return None, 0.0

        try:
            # Preprocess image with source tracking
            processed_image = self.preprocess_image(image, source)

            # Make prediction with error handling
            predictions = self.model.predict(processed_image, verbose=0)

            # Extract results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]

            # Get all class probabilities for debugging
            all_probs = {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}

            print(f"Prediction from {source}: {predicted_class} ({confidence:.3f})")
            print(f"All probabilities: {all_probs}")

            return predicted_class, confidence

        except Exception as e:
            print(f"Error during prediction from {source}: {e}")
            return None, 0.0

    def handle_prediction_result(self, predicted_class, confidence, image_name, source="unknown"):
        """Unified handler for prediction results across all features"""
        if not predicted_class:
            return False

        try:
            # Log prediction
            self.log_prediction(image_name, predicted_class, confidence)

            # Voice announcement with TTS check
            self.speak_prediction(predicted_class, confidence)

            # Debug information
            print(f"Prediction handled - Source: {source}, Class: {predicted_class}, "
                  f"Confidence: {confidence:.3f}, Logged: {image_name}")

            return True

        except Exception as e:
            print(f"Error handling prediction result from {source}: {e}")
            return False

    def get_confidence_message(self, confidence):
        """Get appropriate confidence message based on confidence level"""
        if confidence > 0.9:
            return "✅ Excellent confidence", "success"
        elif confidence > 0.8:
            return "✅ High confidence", "success"
        elif confidence > 0.6:
            return "ℹ️ Moderate confidence", "info"
        else:
            return "⚠️ Low confidence", "warning"
    
    def log_prediction(self, image_name, predicted_class, confidence):
        """Log prediction to history"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_name': image_name,
            'predicted_class': predicted_class,
            'confidence': confidence
        }
        
        # Load existing logs
        logs = []
        if os.path.exists(self.predictions_log):
            try:
                with open(self.predictions_log, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new log
        logs.append(log_entry)
        
        # Save logs
        with open(self.predictions_log, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_recyclability_tip(self, class_name):
        """Get eco-friendly tip for the predicted class"""
        tips = {
            'cardboard': "📦 Cardboard is 100% recyclable! Remove tape and flatten before recycling.",
            'glass': "🥃 Glass can be recycled indefinitely! Rinse containers before disposal.",
            'metal': "🔩 Metal is highly recyclable! Aluminum cans can be recycled in 60 days.",
            'paper': "📄 Paper can be recycled 5-7 times! Remove staples and plastic coatings.",
            'plastic': "🥤 Check the recycling number! Not all plastics are recyclable in all areas.",
            'trash': "🗑️ Consider if this item can be reused, repaired, or properly disposed of."
        }
        return tips.get(class_name, "♻️ Remember to reduce, reuse, and recycle!")

    def validate_camera_image(self, image):
        """Validate camera image quality and provide feedback"""
        try:
            # Convert PIL image to numpy array for analysis
            img_array = np.array(image)

            # Check image dimensions
            height, width = img_array.shape[:2]
            if height < 100 or width < 100:
                return False, "Image too small. Please capture a larger image."

            # Check if image is too dark
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 30:
                return False, "Image too dark. Please ensure better lighting."

            # Check if image is too bright (overexposed)
            if mean_brightness > 220:
                return False, "Image too bright. Please reduce lighting or adjust camera."

            # Check for blur (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                return False, "Image appears blurry. Please keep camera steady."

            return True, "Image quality is good for classification."

        except Exception as e:
            return False, f"Error validating image: {str(e)}"

    def get_camera_tips(self, predicted_class=None, confidence=None):
        """Get camera-specific tips based on prediction results"""
        tips = []

        # General camera tips
        tips.append("📷 Keep the camera steady and ensure good lighting")
        tips.append("🎯 Center the object in the frame for better accuracy")

        # Confidence-based tips
        if confidence is not None:
            if confidence < 0.7:
                tips.append("⚠️ Low confidence - try different angle or better lighting")
                tips.append("🔄 Consider re-taking the photo with clearer view")
            elif confidence > 0.9:
                tips.append("✅ Excellent confidence - great photo quality!")

        # Class-specific tips
        if predicted_class:
            class_tips = {
                'cardboard': "📦 Flatten cardboard for clearer identification",
                'glass': "🥃 Clean glass items show better classification results",
                'metal': "🔩 Metallic surfaces may reflect light - adjust angle",
                'paper': "📄 Spread paper items flat for better recognition",
                'plastic': "🥤 Clear plastic containers work best when clean",
                'trash': "🗑️ Mixed waste items may be harder to classify"
            }
            if predicted_class in class_tips:
                tips.append(class_tips[predicted_class])

        return tips

    def log_camera_session(self, session_stats):
        """Log camera session statistics"""
        try:
            session_log = {
                'timestamp': datetime.now().isoformat(),
                'session_type': 'camera',
                'photos_taken': session_stats.get('photos_taken', 0),
                'successful_predictions': session_stats.get('successful_predictions', 0),
                'average_confidence': session_stats.get('average_confidence', 0.0),
                'most_common_class': session_stats.get('most_common_class', 'none')
            }

            # Load existing session logs
            session_log_path = 'camera_sessions.json'
            logs = []
            if os.path.exists(session_log_path):
                try:
                    with open(session_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []

            # Add new session log
            logs.append(session_log)

            # Keep only last 100 sessions
            if len(logs) > 100:
                logs = logs[-100:]

            # Save logs
            with open(session_log_path, 'w') as f:
                json.dump(logs, f, indent=2)

            return True
        except Exception as e:
            print(f"Error logging camera session: {e}")
            return False

# Initialize classifier with session-aware TTS management
@st.cache_resource
def get_classifier():
    return GarbageClassifier()

def get_classifier_with_fresh_tts():
    """Get classifier and ensure TTS is properly initialized"""
    classifier = get_classifier()
    # Ensure TTS engine is working for this session
    classifier._ensure_tts_engine()
    return classifier

# Use session-aware classifier
classifier = get_classifier_with_fresh_tts()

# Main App
def main():
    """
    Main application function for the Smart Garbage Classification System.

    Developed by Fenil - AI-powered waste classification for environmental sustainability.
    """
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🗂️ Smart Garbage Classification System</h1>
        <p>AI-Powered Waste Classification for a Cleaner Tomorrow</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Load model status with loading indicator
        with st.spinner("Loading model..."):
            model_loaded = classifier.load_model()

        if model_loaded:
            st.success("✅ Model Loaded Successfully!")
        else:
            st.warning("⚠️ No trained model found. Please train a model first.")
        
        # Audio control
        st.subheader("🔊 Audio Settings")

        # Check if TTS is available
        if not classifier.tts_engine:
            st.info("🔇 Voice output not available in cloud environment")
            classifier.mute = True
        else:
            classifier.mute = st.checkbox("🔇 Mute Voice Output", value=classifier.mute)
        
        # Language selection (simplified to English only)
        language = st.selectbox("🌐 Voice Language", ["English"])

        # Update classifier language when changed
        if language != classifier.current_language:
            classifier.set_language(language)
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.radio("Select Page", [
            "🏠 Home",
            "🎯 Train Model", 
            "🔍 Predict Image",
            "📊 Dataset Analysis",
            "📈 Model Performance",
            "📱 Live Camera",
            "📋 Prediction History",
            "📄 Generate Report"
        ])
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Train Model":
        show_training_page()
    elif page == "🔍 Predict Image":
        show_prediction_page()
    elif page == "📊 Dataset Analysis":
        show_dataset_analysis()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "📱 Live Camera":
        show_live_camera()
    elif page == "📋 Prediction History":
        show_prediction_history()
    elif page == "📄 Generate Report":
        show_report_generation()

def show_home_page():
    """Home page with overview"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 AI Classification</h3>
            <p>Advanced deep learning model for accurate waste classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>♻️ Eco-Friendly</h3>
            <p>Promoting sustainable waste management practices</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌍 Multilingual</h3>
            <p>Voice support in multiple languages English, Hindi, Gujarati</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature overview
    st.subheader("🚀 Key Features")
    features = [
        "🤖 Advanced CNN-based garbage classification",
        "🔊 Text-to-speech announcements in multiple languages",
        "📊 Real-time training progress visualization",
        "📈 Comprehensive model performance analysis",
        "📱 Live camera detection capability",
        "📋 Prediction history and logging",
        "📄 Automated PDF report generation",
        "♻️ Eco-friendly recycling tips",
        "🎨 Modern, responsive user interface",
        "☁️ Easy deployment options"
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.write(feature)
    
    st.markdown("---")
    st.info("💡 **Getting Started**: Navigate to 'Train Model' to begin, or 'Predict Image' if you already have a trained model!")

def show_training_page():
    """Model training page"""
    st.header("🎯 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Dataset Selection")
        
        # File uploader for dataset
        uploaded_files = st.file_uploader(
            "Upload your dataset (ZIP file containing folders for each class)",
            type=['zip'],
            help="Upload a ZIP file with folders named: cardboard, glass, metal, paper, plastic, trash"
        )
        
        if uploaded_files:
            # Extract and process dataset
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP file
                with zipfile.ZipFile(uploaded_files, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find dataset structure
                dataset_path = temp_dir
                class_folders = [f for f in os.listdir(dataset_path) 
                               if os.path.isdir(os.path.join(dataset_path, f))]
                
                st.success(f"✅ Dataset extracted! Found {len(class_folders)} classes: {', '.join(class_folders)}")
                
                # Training parameters
                st.subheader("⚙️ Training Parameters")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    epochs = st.slider("Epochs", 5, 50, 20)
                    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
                
                with col_b:
                    validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2)
                    learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.00001], index=0)
                
                # Start training button
                if st.button("🚀 Start Training", type="primary"):
                    # Show initial loading indicator
                    initial_loading = st.empty()
                    with initial_loading.container():
                        st.markdown("""
                        <div class="loading-container">
                            <div class="loading-spinner"></div>
                            <div class="loading-text">
                                Preparing training environment
                                <span class="pulse-dot"></span>
                                <span class="pulse-dot"></span>
                                <span class="pulse-dot"></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    train_model_with_progress(dataset_path, epochs, batch_size, validation_split, learning_rate, initial_loading)
    
    with col2:
        st.subheader("📋 Training Tips")
        st.info("""
        **Dataset Structure:**
        ```
        dataset/
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
        ```
        
        **Recommendations:**
        - Use at least 100 images per class
        - Ensure balanced dataset
        - Images should be clear and well-lit
        - Various angles and conditions
        """)

def train_model_with_progress(dataset_path, epochs, batch_size, validation_split, learning_rate, initial_loading=None):
    """Train model with real-time progress"""

    # Clear initial loading indicator
    if initial_loading:
        initial_loading.empty()

    # Create progress containers
    progress_container = st.container()

    with progress_container:
        st.markdown("""
        <div class="progress-container">
            <h3>🔄 Training Progress</h3>
        </div>
        """, unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training metrics placeholders
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            epoch_metric = st.metric("Epoch", "0/0")
        with col2:
            loss_metric = st.metric("Loss", "0.0")
        with col3:
            acc_metric = st.metric("Accuracy", "0.0%")
        with col4:
            val_acc_metric = st.metric("Val Accuracy", "0.0%")
    
    # Prepare data generators
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split
    )
    
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create and compile model
    model = classifier.create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Custom callback for progress updates
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            # Update status
            status_text.text(f"Epoch {epoch + 1}/{epochs} - Training...")
            
            # Update metrics
            epoch_metric.metric("Epoch", f"{epoch + 1}/{epochs}")
            loss_metric.metric("Loss", f"{logs.get('loss', 0):.4f}")
            acc_metric.metric("Accuracy", f"{logs.get('accuracy', 0)*100:.1f}%")
            val_acc_metric.metric("Val Accuracy", f"{logs.get('val_accuracy', 0)*100:.1f}%")
    
    # Train model
    try:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[ProgressCallback()],
            verbose=0
        )
        
        # Save model
        model.save(classifier.model_path)
        classifier.model = model
        
        # Save history
        with open(classifier.history_path, 'wb') as f:
            pickle.dump(history.history, f)
        
        # Success message
        progress_bar.progress(1.0)
        status_text.text("✅ Training completed successfully!")
        st.success("🎉 Model trained and saved successfully!")
        
        # Show final metrics
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        st.success(f"Final Training Accuracy: {final_acc*100:.2f}%")
        st.success(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
        
        # Voice announcement
        training_message = f"Model training completed with {final_val_acc*100:.1f}% validation accuracy"
        classifier.speak(training_message)
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")

def show_prediction_page():
    """Image prediction page"""
    st.header("🔍 Image Classification")
    
    if not classifier.model:
        st.warning("⚠️ Please train a model first or ensure a trained model is available.")
        return
    
    # Image upload options
    st.subheader("📤 Upload Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of garbage for classification"
        )
        
        # Drag and drop simulation
        if not uploaded_file:
            st.info("📁 Drag and drop an image file here, or click to browse")
    
    with col2:
        st.subheader("📋 Supported Classes")
        for class_name in classifier.class_names:
            st.write(f"• {class_name.title()}")
    
    # Process uploaded image
    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Uploaded Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🎯 Prediction Results")
            
            # Predict button
            if st.button("🔍 Classify Image", type="primary"):
                # Professional loading indicator
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st.markdown("""
                    <div class="loading-container">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">
                            Analyzing image with AI
                            <span class="pulse-dot"></span>
                            <span class="pulse-dot"></span>
                            <span class="pulse-dot"></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Make prediction with source tracking
                predicted_class, confidence = classifier.predict_image(image, "file_upload")

                # Clear loading indicator
                loading_placeholder.empty()

                if predicted_class:
                    # Use unified prediction handler
                    success = classifier.handle_prediction_result(
                        predicted_class, confidence, uploaded_file.name, "file_upload"
                    )

                    if success:
                        # Display results with consistent formatting
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>🏷️ Predicted Class: <strong>{predicted_class.title()}</strong></h3>
                            <h4>📊 Confidence: <strong>{confidence*100:.2f}%</strong></h4>
                        </div>
                        """, unsafe_allow_html=True)

                        # Confidence bar
                        st.progress(confidence)

                        # Confidence message
                        conf_msg, conf_type = classifier.get_confidence_message(confidence)
                        if conf_type == "success":
                            st.success(f"{conf_msg} classification: {predicted_class.title()}")
                        elif conf_type == "info":
                            st.info(f"{conf_msg} classification: {predicted_class.title()}")
                        else:
                            st.warning(f"{conf_msg} classification: {predicted_class.title()}")

                        # Recyclability tip
                        tip = classifier.get_recyclability_tip(predicted_class)
                        st.info(tip)

                        st.success("🎉 Classification completed!")
                    else:
                        st.error("❌ Error processing prediction results")
                else:
                    st.error("❌ Classification failed. Please try again.")

def show_dataset_analysis():
    """Dataset statistics and analysis"""
    st.header("📊 Dataset Analysis")

    # Show loading indicator while preparing analysis
    with st.spinner("Analyzing dataset..."):
        # Sample dataset statistics (you can modify this to read from actual dataset)
        sample_data = {
            'cardboard': 500,
            'glass': 420,
            'metal': 380,
            'paper': 450,
            'plastic': 480,
            'trash': 370
        }
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Class Distribution")
        
        # Bar chart
        fig_bar = px.bar(
            x=list(sample_data.keys()),
            y=list(sample_data.values()),
            title="Images per Class",
            labels={'x': 'Garbage Class', 'y': 'Number of Images'}
        )
        fig_bar.update_traces(marker_color='#4CAF50')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Dataset Balance")
        
        # Pie chart
        fig_pie = px.pie(
            values=list(sample_data.values()),
            names=list(sample_data.keys()),
            title="Class Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Dataset statistics
    st.subheader("📋 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_images = sum(sample_data.values())
    avg_per_class = total_images / len(sample_data)
    max_class = max(sample_data, key=sample_data.get)
    min_class = min(sample_data, key=sample_data.get)
    
    with col1:
        st.metric("Total Images", f"{total_images:,}")
    with col2:
        st.metric("Avg per Class", f"{avg_per_class:.0f}")
    with col3:
        st.metric("Most Common", f"{max_class} ({sample_data[max_class]})")
    with col4:
        st.metric("Least Common", f"{min_class} ({sample_data[min_class]})")
    
    # Balance warning
    imbalance_ratio = sample_data[max_class] / sample_data[min_class]
    if imbalance_ratio > 1.5:
        st.warning(f"⚠️ Dataset imbalance detected! {max_class} has {imbalance_ratio:.1f}x more images than {min_class}")
    else:
        st.success("✅ Dataset appears balanced!")

def show_model_performance():
    """Model performance metrics and analysis"""
    st.header("📈 Model Performance Analysis")

    if not os.path.exists(classifier.history_path):
        st.warning("⚠️ No training history found. Train a model first to see performance metrics.")
        return

    # Load training history with loading indicator
    with st.spinner("Loading performance metrics..."):
        try:
            with open(classifier.history_path, 'rb') as f:
                history = pickle.load(f)
        except:
            st.error("❌ Could not load training history.")
            return
    
    # Training curves
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Training & Validation Loss")
        
        epochs = range(1, len(history['loss']) + 1)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=list(epochs), y=history['loss'], 
                                    mode='lines', name='Training Loss'))
        fig_loss.add_trace(go.Scatter(x=list(epochs), y=history['val_loss'], 
                                    mode='lines', name='Validation Loss'))
        fig_loss.update_layout(title="Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        st.subheader("📈 Training & Validation Accuracy")
        
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=list(epochs), y=history['accuracy'], 
                                   mode='lines', name='Training Accuracy'))
        fig_acc.add_trace(go.Scatter(x=list(epochs), y=history['val_accuracy'], 
                                   mode='lines', name='Validation Accuracy'))
        fig_acc.update_layout(title="Accuracy Over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy")
        st.plotly_chart(fig_acc, use_container_width=True)
    
    # Final metrics
    st.subheader("🎯 Final Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    with col1:
        st.metric("Training Accuracy", f"{final_train_acc*100:.2f}%")
    with col2:
        st.metric("Validation Accuracy", f"{final_val_acc*100:.2f}%")
    with col3:
        st.metric("Training Loss", f"{final_train_loss:.4f}")
    with col4:
        st.metric("Validation Loss", f"{final_val_loss:.4f}")
    
    # Model evaluation
    if abs(final_train_acc - final_val_acc) > 0.1:
        st.warning("⚠️ Possible overfitting detected. Consider adding more regularization or data.")
    else:
        st.success("✅ Model shows good generalization!")

def show_live_camera():
    """Live camera detection with real-time classification"""
    st.header("📱 Live Camera Detection")

    # Check if model is loaded
    if not classifier.model:
        st.warning("⚠️ Please train a model first or ensure a trained model is available.")
        return

    # Device compatibility and error handling
    try:
        # Check if running in a supported environment
        import platform
        system_info = platform.system()

        # Display system compatibility info
        with st.expander("🔧 System Information", expanded=False):
            st.info(f"""
            **System:** {system_info}
            **Camera Support:** Streamlit camera_input widget
            **Browser Required:** Modern browser with camera access
            **Permissions:** Camera access must be granted
            """)

        # Camera access instructions
        if st.session_state.get('show_camera_help', False):
            st.info("""
            📷 **Camera Access Help:**
            1. Click the camera button below to activate your camera
            2. Grant camera permissions when prompted by your browser
            3. Position the garbage item in the camera view
            4. Click the capture button to take a photo
            5. The system will automatically classify the captured image
            """)

            if st.button("✅ Got it, hide help"):
                st.session_state.show_camera_help = False
                st.rerun()
        else:
            if st.button("❓ Show Camera Help"):
                st.session_state.show_camera_help = True
                st.rerun()

    except Exception as e:
        st.error(f"❌ System compatibility check failed: {str(e)}")
        st.info("💡 Please ensure you're using a modern browser with camera support")

    # Initialize session state for camera
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'camera_capture' not in st.session_state:
        st.session_state.camera_capture = None
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = 0.0
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0

    # Camera interface layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Camera Feed")

        # Camera troubleshooting section
        if st.session_state.get('camera_error', False):
            st.error("""
            ❌ **Camera Access Issues Detected**

            **Common Solutions:**
            - Ensure camera permissions are granted in your browser
            - Check if another application is using the camera
            - Try refreshing the page and granting permissions again
            - Ensure you're using HTTPS (required for camera access)
            """)

            if st.button("🔄 Try Camera Again"):
                st.session_state.camera_error = False
                st.rerun()

        # Camera input widget with error handling
        try:
            camera_photo = st.camera_input(
                "Take a photo for garbage classification",
                key="garbage_camera",
                help="Position the garbage item in the camera view and take a photo for classification"
            )

            # Reset error state if camera works
            if 'camera_error' in st.session_state:
                st.session_state.camera_error = False

        except Exception as e:
            st.session_state.camera_error = True
            st.error(f"❌ Camera initialization failed: {str(e)}")
            st.info("""
            💡 **Troubleshooting Tips:**
            - Refresh the page and try again
            - Check browser camera permissions
            - Ensure no other apps are using the camera
            - Try a different browser if issues persist
            """)
            camera_photo = None

        # Process captured image
        if camera_photo is not None:
            # Display captured image
            image = Image.open(camera_photo)
            st.image(image, caption="Captured Image", use_column_width=True)

            # Validate image quality
            is_valid, validation_message = classifier.validate_camera_image(image)

            if not is_valid:
                st.warning(f"⚠️ Image Quality Issue: {validation_message}")
                st.info("💡 Try taking another photo with better conditions")

            # Auto-classify the captured image
            with st.spinner("� Analyzing captured image..."):
                try:
                    predicted_class, confidence = classifier.predict_image(image, "live_camera")

                    if predicted_class:
                        # Update session state
                        st.session_state.last_prediction = predicted_class
                        st.session_state.last_confidence = confidence
                        st.session_state.prediction_count += 1

                        # Use unified prediction handler
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_name = f"camera_capture_{timestamp}.jpg"

                        success = classifier.handle_prediction_result(
                            predicted_class, confidence, image_name, "live_camera"
                        )

                        if success:
                            # Show success message with confidence level
                            conf_msg, conf_type = classifier.get_confidence_message(confidence)
                            if conf_type == "success":
                                st.success(f"{conf_msg} classification: {predicted_class.title()}")
                            elif conf_type == "info":
                                st.info(f"{conf_msg} classification: {predicted_class.title()}")
                            else:
                                st.warning(f"{conf_msg} classification: {predicted_class.title()}")
                        else:
                            st.error("❌ Error processing prediction results")
                    else:
                        st.error("❌ Classification failed. Please try again with a clearer image.")

                except Exception as e:
                    st.error(f"❌ Error during classification: {str(e)}")
                    st.info("💡 Please try taking another photo")

    with col2:
        st.subheader("🎛️ Camera Controls")

        # Camera status
        if camera_photo is not None:
            st.success("📸 Image captured successfully!")
        else:
            st.info("� Ready to capture - Click the camera button above")

        # Manual classification button for current capture
        if camera_photo is not None:
            if st.button("🔍 Re-analyze Image", type="primary"):
                with st.spinner("🔍 Re-analyzing image..."):
                    try:
                        predicted_class, confidence = classifier.predict_image(image, "camera_reanalysis")

                        if predicted_class:
                            st.session_state.last_prediction = predicted_class
                            st.session_state.last_confidence = confidence
                            st.session_state.prediction_count += 1

                            # Use unified prediction handler
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_name = f"camera_reanalysis_{timestamp}.jpg"

                            success = classifier.handle_prediction_result(
                                predicted_class, confidence, image_name, "camera_reanalysis"
                            )

                            if success:
                                st.success("🔄 Image re-analyzed successfully!")
                                # Show confidence message
                                conf_msg, conf_type = classifier.get_confidence_message(confidence)
                                if conf_type == "success":
                                    st.success(f"{conf_msg}: {predicted_class.title()}")
                                elif conf_type == "info":
                                    st.info(f"{conf_msg}: {predicted_class.title()}")
                                else:
                                    st.warning(f"{conf_msg}: {predicted_class.title()}")
                            else:
                                st.error("❌ Error processing re-analysis results")
                        else:
                            st.error("❌ Re-analysis failed. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error during re-analysis: {str(e)}")

            # Get camera-specific tips
            if st.session_state.last_prediction and st.session_state.last_confidence:
                camera_tips = classifier.get_camera_tips(
                    st.session_state.last_prediction,
                    st.session_state.last_confidence
                )

                if st.button("💡 Get Photography Tips"):
                    st.info("📸 **Camera Tips:**")
                    for tip in camera_tips[:3]:  # Show first 3 tips
                        st.write(f"• {tip}")

        # Clear capture button
        if camera_photo is not None:
            if st.button("🗑️ Clear Capture"):
                st.session_state.camera_capture = None
                st.rerun()

        # Session management
        st.markdown("---")
        st.subheader("🔧 Session Management")

        col_session = st.columns(2)
        with col_session[0]:
            if st.button("📊 Save Session Stats"):
                if st.session_state.prediction_count > 0:
                    session_stats = {
                        'photos_taken': st.session_state.prediction_count,
                        'successful_predictions': st.session_state.prediction_count,
                        'average_confidence': st.session_state.last_confidence,
                        'most_common_class': st.session_state.last_prediction or 'none'
                    }

                    if classifier.log_camera_session(session_stats):
                        st.success("✅ Session stats saved!")
                    else:
                        st.error("❌ Failed to save session stats")
                else:
                    st.info("📸 No photos taken in this session")

        with col_session[1]:
            if st.button("🔄 Reset Session"):
                st.session_state.prediction_count = 0
                st.session_state.last_prediction = None
                st.session_state.last_confidence = 0.0
                st.success("🔄 Session reset!")
                st.rerun()

        st.markdown("---")

        # Live Results Section
        st.subheader("📊 Latest Results")

        if st.session_state.last_prediction:
            # Display latest prediction with styling
            st.markdown(f"""
            <div class="prediction-card">
                <h4>🏷️ Detected Class: <strong>{st.session_state.last_prediction.title()}</strong></h4>
                <h5>📊 Confidence: <strong>{st.session_state.last_confidence*100:.2f}%</strong></h5>
            </div>
            """, unsafe_allow_html=True)

            # Confidence progress bar
            st.progress(st.session_state.last_confidence)

            # Recyclability tip
            tip = classifier.get_recyclability_tip(st.session_state.last_prediction)
            st.info(tip)

        else:
            st.write("**Detected Class:** None")
            st.write("**Confidence:** 0.0%")
            st.info("📸 Take a photo to see classification results")

        # Session statistics
        st.markdown("---")
        st.subheader("📈 Session Stats")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Photos Taken", st.session_state.prediction_count)
        with col_b:
            if st.session_state.last_confidence > 0:
                st.metric("Last Confidence", f"{st.session_state.last_confidence*100:.1f}%")
            else:
                st.metric("Last Confidence", "0.0%")

    # Additional features section
    st.markdown("---")
    st.subheader("✨ Camera Features")

    feature_cols = st.columns(3)

    with feature_cols[0]:
        st.markdown("""
        **📸 Instant Classification**
        - Automatic analysis on capture
        - Real-time confidence scoring
        - Voice announcements
        """)

    with feature_cols[1]:
        st.markdown("""
        **📊 Smart Logging**
        - Automatic prediction logging
        - Timestamp tracking
        - History integration
        """)

    with feature_cols[2]:
        st.markdown("""
        **♻️ Eco-Friendly Tips**
        - Recycling guidance
        - Sustainability advice
        - Environmental impact
        """)

    # Tips for better results
    st.markdown("---")
    st.subheader("� Tips for Better Results")

    tips_cols = st.columns(2)

    with tips_cols[0]:
        st.info("""
        **📷 Photography Tips:**
        - Ensure good lighting
        - Center the object in frame
        - Avoid cluttered backgrounds
        - Keep camera steady
        """)

    with tips_cols[1]:
        st.info("""
        **🎯 Classification Tips:**
        - Use clear, single objects
        - Avoid mixed materials
        - Clean objects work better
        - Try different angles if needed
        """)

def show_prediction_history():
    """Show prediction history and logs"""
    st.header("📋 Prediction History")

    if not os.path.exists(classifier.predictions_log):
        st.info("📝 No predictions logged yet. Make some predictions to see history!")
        return

    # Load prediction logs with loading indicator
    with st.spinner("Loading prediction history..."):
        try:
            with open(classifier.predictions_log, 'r') as f:
                logs = json.load(f)
        except:
            st.error("❌ Could not load prediction history.")
            return
    
    if not logs:
        st.info("📝 No predictions logged yet.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['confidence'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    
    # Display logs
    st.subheader(f"📊 Total Predictions: {len(logs)}")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_common = df['predicted_class'].mode().iloc[0] if not df.empty else "None"
        st.metric("Most Common Class", most_common)
    
    with col2:
        avg_confidence = df['confidence'].str.rstrip('%').astype(float).mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    with col3:
        recent_predictions = len(df[df['timestamp'] > datetime.now() - pd.Timedelta(days=1)])
        st.metric("Today's Predictions", recent_predictions)
    
    # Display table
    st.subheader("📋 Recent Predictions")
    st.dataframe(
        df[['timestamp', 'image_name', 'predicted_class', 'confidence']].sort_values('timestamp', ascending=False),
        use_container_width=True
    )
    
    # Class distribution chart
    st.subheader("📊 Classification Distribution")
    class_counts = df['predicted_class'].value_counts()
    
    fig = px.bar(
        x=class_counts.index,
        y=class_counts.values,
        title="Predictions by Class",
        labels={'x': 'Garbage Class', 'y': 'Number of Predictions'}
    )
    fig.update_traces(marker_color='#4CAF50')
    st.plotly_chart(fig, use_container_width=True)
    
    # Clear history option
    if st.button("🗑️ Clear History", type="secondary"):
        if st.checkbox("⚠️ Confirm deletion"):
            os.remove(classifier.predictions_log)
            st.success("✅ History cleared!")
            st.experimental_rerun()

def show_report_generation():
    """Generate and download PDF reports"""
    st.header("📄 Generate Report")
    
    st.subheader("📋 Report Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Model Performance Report", "Prediction History Report", "Complete Analysis Report"]
        )
        
        include_charts = st.checkbox("📊 Include Charts", value=True)
        include_tips = st.checkbox("♻️ Include Recycling Tips", value=True)
    
    with col2:
        st.subheader("📝 Report Preview")
        st.info(f"""
        **Selected Report:** {report_type}
        **Include Charts:** {'Yes' if include_charts else 'No'}
        **Include Tips:** {'Yes' if include_tips else 'No'}
        """)
    
    # Generate report button
    if st.button("📄 Generate PDF Report", type="primary"):
        # Professional loading indicator for report generation
        report_loading = st.empty()
        with report_loading.container():
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">
                    Generating PDF report
                    <span class="pulse-dot"></span>
                    <span class="pulse-dot"></span>
                    <span class="pulse-dot"></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        pdf_buffer = generate_pdf_report(report_type, include_charts, include_tips)

        # Clear loading indicator
        report_loading.empty()

        if pdf_buffer:
            st.success("✅ Report generated successfully!")

            # Download button
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"garbage_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

            classifier.speak("Report generated successfully")
        else:
            st.error("❌ Failed to generate report")

def generate_pdf_report(report_type, include_charts, include_tips):
    """Generate PDF report"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph(f"<b>Smart Garbage Classification System</b><br/>{report_type}", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Report content based on type
        if report_type == "Model Performance Report":
            # Model info
            story.append(Paragraph("<b>Model Information</b>", styles['Heading2']))
            story.append(Paragraph("Architecture: Convolutional Neural Network (CNN)", styles['Normal']))
            story.append(Paragraph("Classes: cardboard, glass, metal, paper, plastic, trash", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Performance metrics (if available)
            if os.path.exists(classifier.history_path):
                try:
                    with open(classifier.history_path, 'rb') as f:
                        history = pickle.load(f)
                    
                    final_acc = history['accuracy'][-1]
                    final_val_acc = history['val_accuracy'][-1]
                    
                    story.append(Paragraph("<b>Performance Metrics</b>", styles['Heading2']))
                    story.append(Paragraph(f"Training Accuracy: {final_acc*100:.2f}%", styles['Normal']))
                    story.append(Paragraph(f"Validation Accuracy: {final_val_acc*100:.2f}%", styles['Normal']))
                    story.append(Spacer(1, 12))
                except:
                    pass
        
        elif report_type == "Prediction History Report":
            # Prediction statistics
            if os.path.exists(classifier.predictions_log):
                try:
                    with open(classifier.predictions_log, 'r') as f:
                        logs = json.load(f)
                    
                    story.append(Paragraph("<b>Prediction Statistics</b>", styles['Heading2']))
                    story.append(Paragraph(f"Total Predictions: {len(logs)}", styles['Normal']))
                    
                    if logs:
                        df = pd.DataFrame(logs)
                        most_common = df['predicted_class'].mode().iloc[0]
                        avg_confidence = df['confidence'].mean()
                        
                        story.append(Paragraph(f"Most Common Class: {most_common}", styles['Normal']))
                        story.append(Paragraph(f"Average Confidence: {avg_confidence*100:.1f}%", styles['Normal']))
                    
                    story.append(Spacer(1, 12))
                except:
                    pass
        
        # Recycling tips
        if include_tips:
            story.append(Paragraph("<b>Recycling Tips</b>", styles['Heading2']))
            tips = {
                'Cardboard': "Remove tape and flatten before recycling. 100% recyclable!",
                'Glass': "Rinse containers before disposal. Can be recycled indefinitely!",
                'Metal': "Highly recyclable. Aluminum cans can be recycled in 60 days.",
                'Paper': "Remove staples and plastic coatings. Can be recycled 5-7 times.",
                'Plastic': "Check recycling numbers. Not all plastics are recyclable everywhere.",
                'General': "Remember: Reduce, Reuse, Recycle!"
            }
            
            for category, tip in tips.items():
                story.append(Paragraph(f"<b>{category}:</b> {tip}", styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # Footer
        story.append(Paragraph(f"<i>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['Normal']))
        story.append(Paragraph("<i>Smart Garbage Classification System - AI for a Cleaner Tomorrow</i>", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None



# Main execution
if __name__ == "__main__":
    main()

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><small>🌱 Built for a sustainable future</small></p>
        <p><small>♻️ AI-Powered Waste Classification</small></p>
        <p><small><strong>Developed by Fenil</strong></small></p>
        <p><small>v1.0.0 | MIT License</small></p>
    </div>
    """, unsafe_allow_html=True)