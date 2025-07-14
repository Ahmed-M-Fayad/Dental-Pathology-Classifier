import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Dental Pathology Classification System",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for premium medical appearance with improved dark mode
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for theming */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fbff;
        --bg-sidebar: #f0f4f8;
        --text-primary: #1a202c;
        --text-secondary: #4a5568;
        --border-color: #e2e8f0;
        --shadow-light: rgba(0,0,0,0.05);
        --shadow-medium: rgba(0,0,0,0.1);
        --shadow-heavy: rgba(0,0,0,0.15);
        --card-bg: #ffffff;
        --accent-primary: #3182ce;
        --accent-secondary: #2b6cb0;
        --success: #38a169;
        --warning: #d69e2e;
        --error: #e53e3e;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-success: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        --gradient-warning: linear-gradient(135deg, #ed8936 0%, #d69e2e 100%);
        --gradient-error: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
    }
    
    [data-theme="dark"] {
        --bg-primary: #0f1419;
        --bg-secondary: #1a1f29;
        --bg-sidebar: #252a35;
        --text-primary: #f7fafc;
        --text-secondary: #a0aec0;
        --border-color: #2d3748;
        --shadow-light: rgba(255,255,255,0.02);
        --shadow-medium: rgba(255,255,255,0.05);
        --shadow-heavy: rgba(255,255,255,0.08);
        --card-bg: #1a1f29;
        --accent-primary: #63b3ed;
        --accent-secondary: #4299e1;
        --success: #68d391;
        --warning: #f6e05e;
        --error: #fc8181;
        --gradient-primary: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        --gradient-success: linear-gradient(135deg, #68d391 0%, #48bb78 100%);
        --gradient-warning: linear-gradient(135deg, #f6e05e 0%, #ed8936 100%);
        --gradient-error: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f1419;
            --bg-secondary: #1a1f29;
            --bg-sidebar: #252a35;
            --text-primary: #f7fafc;
            --text-secondary: #a0aec0;
            --border-color: #2d3748;
            --shadow-light: rgba(255,255,255,0.02);
            --shadow-medium: rgba(255,255,255,0.05);
            --shadow-heavy: rgba(255,255,255,0.08);
            --card-bg: #1a1f29;
            --accent-primary: #63b3ed;
            --accent-secondary: #4299e1;
            --success: #68d391;
            --warning: #f6e05e;
            --error: #fc8181;
            --gradient-primary: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            --gradient-success: linear-gradient(135deg, #68d391 0%, #48bb78 100%);
            --gradient-warning: linear-gradient(135deg, #f6e05e 0%, #ed8936 100%);
            --gradient-error: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
        }
    }
    
    /* Base styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .sub-header {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* Card components */
    .prediction-card {
        background: var(--gradient-primary);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px var(--shadow-medium);
        border: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, rgba(255,255,255,0.3), rgba(255,255,255,0.1), rgba(255,255,255,0.3));
    }
    
    .prediction-card h2 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prediction-card h3 {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    /* Risk level cards */
    .risk-high {
        background: var(--gradient-error);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(229, 62, 62, 0.2);
        border: 1px solid rgba(229, 62, 62, 0.3);
    }
    
    .risk-medium {
        background: var(--gradient-warning);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(214, 158, 46, 0.2);
        border: 1px solid rgba(214, 158, 46, 0.3);
    }
    
    .risk-low {
        background: var(--gradient-success);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(56, 161, 105, 0.2);
        border: 1px solid rgba(56, 161, 105, 0.3);
    }
    
    .risk-high h4, .risk-medium h4, .risk-low h4 {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Medical info card */
    .medical-info {
        background: var(--card-bg);
        color: var(--text-primary);
        padding: 2rem;
        border-left: 4px solid var(--accent-primary);
        border-radius: 0 12px 12px 0;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px var(--shadow-light);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .medical-info:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px var(--shadow-medium);
    }
    
    .medical-info h4 {
        color: var(--accent-primary);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .medical-info p {
        margin-bottom: 0.8rem;
        line-height: 1.6;
    }
    
    .medical-info strong {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* Stats cards */
    .stats-card {
        background: var(--card-bg);
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px var(--shadow-light);
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stats-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px var(--shadow-medium);
    }
    
    .stats-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-success);
    }
    
    .stats-card h4 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
    }
    
    .stats-card p {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent-primary);
        margin: 0;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: var(--card-bg);
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px var(--shadow-light);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease;
    }
    
    .sidebar-section:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px var(--shadow-medium);
    }
    
    .sidebar-section h4 {
        color: var(--accent-primary);
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .sidebar-section ul {
        padding-left: 1.2rem;
    }
    
    .sidebar-section li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    .sidebar-section ol {
        padding-left: 1.2rem;
    }
    
    .sidebar-section ol li {
        margin-bottom: 0.6rem;
        line-height: 1.5;
    }
    
    /* Category cards in sidebar */
    .category-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .category-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px var(--shadow-light);
    }
    
    .category-card strong {
        display: block;
        margin-bottom: 0.25rem;
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .category-card small {
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: var(--gradient-primary);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stProgress > div {
        background: var(--bg-secondary);
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    
    /* Metrics styling */
    .stMetric {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px var(--shadow-light);
        transition: transform 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--shadow-medium);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px var(--shadow-light);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px var(--shadow-medium);
        background: var(--gradient-primary);
    }
    
    /* File uploader */
    .stFileUploader {
        background: var(--card-bg);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-primary);
        background: var(--bg-secondary);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--card-bg);
        border-radius: 8px 8px 0 0;
        border: 1px solid var(--border-color);
        border-bottom: none;
        box-shadow: 0 2px 8px var(--shadow-light);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-bottom: none;
        color: var(--text-primary);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--card-bg);
        transform: translateY(-1px);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 0 0 8px 8px;
        padding: 2rem;
        box-shadow: 0 4px 16px var(--shadow-light);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-secondary);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px var(--shadow-light);
    }
    
    .streamlit-expanderContent {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1.5rem;
    }
    
    /* Alert styling */
    .stAlert {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        box-shadow: 0 2px 8px var(--shadow-light);
    }
    
    .stSuccess {
        background: rgba(72, 187, 120, 0.1);
        border: 1px solid var(--success);
        color: var(--success);
    }
    
    .stWarning {
        background: rgba(214, 158, 46, 0.1);
        border: 1px solid var(--warning);
        color: var(--warning);
    }
    
    .stError {
        background: rgba(229, 62, 62, 0.1);
        border: 1px solid var(--error);
        color: var(--error);
    }
    
    .stInfo {
        background: rgba(49, 130, 206, 0.1);
        border: 1px solid var(--accent-primary);
        color: var(--accent-primary);
    }
    
    /* Spinner customization */
    .stSpinner {
        text-align: center;
        color: var(--accent-primary);
    }
    
    /* Image styling */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px var(--shadow-light);
        border: 1px solid var(--border-color);
    }
    
    /* Footer styling */
    .footer-section {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.9rem;
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-top: 2rem;
        box-shadow: 0 4px 16px var(--shadow-light);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
        }
        
        .prediction-card {
            padding: 1.5rem;
        }
        
        .medical-info {
            padding: 1.5rem;
        }
        
        .stats-card {
            padding: 1.5rem;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
    }
</style>
""", unsafe_allow_html=True)

# Class names and comprehensive descriptions
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
CLASS_DESCRIPTIONS = {
    'CaS': 'Cold Sore (Herpes Simplex)',
    'CoS': 'Canker Sore (Aphthous Ulcer)', 
    'Gum': 'Gum Disease (Periodontal Disease)',
    'MC': 'Mucocele (Mucous Cyst)',
    'OC': 'Oral Cancer',
    'OLP': 'Oral Lichen Planus',
    'OT': 'Other/Normal Tissue'
}

# Medical information for each condition
MEDICAL_INFO = {
    'CaS': {
        'description': 'Viral infection causing fluid-filled blisters on lips or around mouth',
        'symptoms': 'Burning sensation, fluid-filled blisters, crusting',
        'risk_level': 'Low',
        'recommendation': 'Usually heals on its own in 7-10 days. Antiviral medications may help.',
        'seek_help': 'If frequent outbreaks or severe symptoms occur'
    },
    'CoS': {
        'description': 'Shallow, painful ulcers inside the mouth',
        'symptoms': 'Round/oval sores with white/gray center and red border',
        'risk_level': 'Low',
        'recommendation': 'Typically heals within 1-2 weeks. Avoid spicy/acidic foods.',
        'seek_help': 'If ulcers are large, persistent, or frequently recurring'
    },
    'Gum': {
        'description': 'Infection and inflammation of gums and supporting structures',
        'symptoms': 'Red, swollen, bleeding gums, bad breath',
        'risk_level': 'Medium',
        'recommendation': 'Requires professional dental treatment and improved oral hygiene.',
        'seek_help': 'See dentist immediately for proper treatment'
    },
    'MC': {
        'description': 'Benign cyst containing mucus, usually on inner lip',
        'symptoms': 'Soft, painless, bluish bump filled with fluid',
        'risk_level': 'Low',
        'recommendation': 'Often resolves spontaneously. Avoid biting or trauma.',
        'seek_help': 'If large, persistent, or causing discomfort'
    },
    'OC': {
        'description': 'Malignant tumor that can occur anywhere in the mouth',
        'symptoms': 'Persistent sore, white/red patches, difficulty swallowing',
        'risk_level': 'High',
        'recommendation': 'IMMEDIATE medical attention required for proper diagnosis and treatment.',
        'seek_help': 'Consult oncologist or oral surgeon immediately'
    },
    'OLP': {
        'description': 'Chronic inflammatory condition affecting oral mucosa',
        'symptoms': 'White lacy patches, red inflamed areas, burning sensation',
        'risk_level': 'Medium',
        'recommendation': 'Requires medical management and regular monitoring.',
        'seek_help': 'See oral medicine specialist for proper treatment'
    },
    'OT': {
        'description': 'Normal healthy oral tissue or benign condition',
        'symptoms': 'No significant pathological changes',
        'risk_level': 'Low',
        'recommendation': 'Maintain good oral hygiene and regular dental checkups.',
        'seek_help': 'Continue routine dental care'
    }
}

# Risk level colors
RISK_COLORS = {
    'Low': '#48bb78',
    'Medium': '#ed8936', 
    'High': '#f56565'
}

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def load_model():
    """Load the trained EfficientNetB0 model with progress tracking"""
    try:
        model_path = "efficientnetb0_transfer_final.keras"
        
        # Create a progress bar for model loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('🔄 Loading model architecture...')
        progress_bar.progress(25)
        
        model = tf.keras.models.load_model(model_path)
        
        status_text.text('✅ Model loaded successfully!')
        progress_bar.progress(100)
        
        time.sleep(0.5)  # Brief pause to show completion
        progress_bar.empty()
        status_text.empty()
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("📁 Please ensure the model file 'efficientnetb0_transfer_final.keras' is in the same directory as this app")
        return None

def preprocess_image(image):
    """Preprocess image for EfficientNetB0 model"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Store original size for display
    original_size = image.size
    
    # Resize to model input size (256x256)
    image = image.resize((256, 256))
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    return img_array, original_size

def predict_condition(model, image):
    """Make prediction on the image with detailed results"""
    try:
        # Preprocess image
        processed_image, original_size = preprocess_image(image)
        
        # Make prediction with progress tracking
        with st.spinner("🔍 Analyzing image..."):
            predictions = model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(CLASS_NAMES):
            class_probabilities[class_name] = predictions[0][i]
        
        # Sort probabilities for top 3
        sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_3_predictions = sorted_probs[:3]
        
        return predicted_class, confidence, class_probabilities, top_3_predictions, original_size
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None, None, None

def display_prediction_results(predicted_class, confidence, top_3_predictions):
    """Display enhanced prediction results"""
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <h2>🎯 Primary Diagnosis</h2>
        <h3>{CLASS_DESCRIPTIONS[predicted_class]}</h3>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk assessment
    risk_level = MEDICAL_INFO[predicted_class]['risk_level']
    risk_class = f"risk-{risk_level.lower()}"
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h4>⚠️ Risk Level: {risk_level}</h4>
        <p>{MEDICAL_INFO[predicted_class]['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 3 predictions
    st.markdown("### 📊 Top 3 Predictions")
    
    for i, (class_name, prob) in enumerate(top_3_predictions):
        rank_emoji = ["🥇", "🥈", "🥉"][i]
        description = CLASS_DESCRIPTIONS[class_name]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{rank_emoji} **{description}**")
            st.progress(float(prob))
        with col2:
            st.metric("Confidence", f"{prob:.1%}")
    
    # Medical details
    st.markdown("### 🏥 Medical Information")
    st.markdown(f"""
    <div class="medical-info">
        <h4>📋 Condition Details</h4>
        <p><strong>Description:</strong> {MEDICAL_INFO[predicted_class]['description']}</p>
        <p><strong>Common Symptoms:</strong> {MEDICAL_INFO[predicted_class]['symptoms']}</p>
        <p><strong>When to Seek Help:</strong> {MEDICAL_INFO[predicted_class]['seek_help']}</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create comprehensive sidebar with model and medical information"""
    
    st.sidebar.markdown("# 🦷 Dental AI Assistant")
    st.sidebar.markdown("---")
    
    # Model Information
    st.sidebar.markdown("## 🤖 Model Information")
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h4>EfficientNetB0 Transfer Learning</h4>
        <ul>
            <li><strong>Architecture:</strong> EfficientNetB0</li>
            <li><strong>Input Size:</strong> 256×256 pixels</li>
            <li><strong>Classes:</strong> 7 oral conditions</li>
            <li><strong>Training Images:</strong> 5,624</li>
            <li><strong>Preprocessing:</strong> ImageNet normalization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Classification Categories
    st.sidebar.markdown("## 📋 Classification Categories")
    for class_code, description in CLASS_DESCRIPTIONS.items():
        risk_level = MEDICAL_INFO[class_code]['risk_level']
        risk_color = RISK_COLORS[risk_level]
        
        st.sidebar.markdown(f"""
        <div class="category-card">
            <strong>{description}</strong>
            <small>Risk Level: <span style="color: {risk_color};">{risk_level}</span></small>
        </div>
        """, unsafe_allow_html=True)
    
    # Usage Instructions
    st.sidebar.markdown("## 📖 Usage Instructions")
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h4>How to Use</h4>
        <ol>
            <li>Upload a clear dental/oral image</li>
            <li>Wait for AI analysis</li>
            <li>Review prediction results</li>
            <li>Follow medical recommendations</li>
            <li>Consult healthcare professionals</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Important Disclaimer
    st.sidebar.markdown("## ⚠️ Important Disclaimer")
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h4>Medical Advisory</h4>
        <p>This AI tool is for educational purposes only and should not replace professional medical diagnosis. Always consult qualified healthcare providers for proper medical evaluation and treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    if st.session_state.prediction_history:
        st.sidebar.markdown("## 📊 Session Statistics")
        total_predictions = len(st.session_state.prediction_history)
        avg_confidence = np.mean([pred['confidence'] for pred in st.session_state.prediction_history])
        most_common = max(set([pred['class'] for pred in st.session_state.prediction_history]), 
                         key=[pred['class'] for pred in st.session_state.prediction_history].count)
        
        st.sidebar.markdown(f"""
        <div class="stats-card">
            <h4>Total Predictions</h4>
            <p>{total_predictions}</p>
        </div>
        <div class="stats-card">
            <h4>Average Confidence</h4>
            <p>{avg_confidence:.1%}</p>
        </div>
        <div class="stats-card">
            <h4>Most Common</h4>
            <p>{CLASS_DESCRIPTIONS[most_common]}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        🦷 Dental Pathology Classification System
    </div>
    <div class="sub-header">
        Advanced AI-powered oral health diagnosis using EfficientNetB0 deep learning
    </div>
    """, unsafe_allow_html=True)
    
    # Create sidebar
    create_sidebar()
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Cannot proceed without model. Please check the model file.")
        return
    
    # File uploader
    st.markdown("## 📤 Upload Dental Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a clear image of the oral condition for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image details
            st.markdown(f"""
            <div class="stats-card">
                <h4>Image Details</h4>
                <p><strong>Format:</strong> {image.format}</p>
                <p><strong>Size:</strong> {image.size[0]}×{image.size[1]}</p>
                <p><strong>Mode:</strong> {image.mode}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Make prediction
            predicted_class, confidence, class_probabilities, top_3_predictions, original_size = predict_condition(model, image)
            
            if predicted_class is not None:
                # Display results
                display_prediction_results(predicted_class, confidence, top_3_predictions)
                
                # Add to history
                st.session_state.prediction_history.append({
                    'class': predicted_class,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Success message
                st.success("✅ Analysis completed successfully!")
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("## 📈 Prediction History")
        
        # Create expandable history section
        with st.expander("View Previous Predictions", expanded=False):
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-10:])):  # Show last 10
                st.markdown(f"""
                <div class="medical-info">
                    <h4>Prediction #{len(st.session_state.prediction_history) - i}</h4>
                    <p><strong>Condition:</strong> {CLASS_DESCRIPTIONS[pred['class']]}</p>
                    <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                    <p><strong>Time:</strong> {pred['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.success("History cleared!")
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-section">
        <p>🏥 <strong>Dental Pathology Classification System</strong></p>
        <p>Powered by EfficientNetB0 Deep Learning | For Educational Use Only</p>
        <p>⚠️ Always consult healthcare professionals for proper medical diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    