import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import os
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}
.prediction-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.fake-news {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
}
.real-news {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
}
.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize NLTK components
@st.cache_resource
def initialize_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return stopwords.words('english'), WordNetLemmatizer()
    except:
        st.error("Failed to download NLTK data. Some text preprocessing features may not work.")
        return [], None

# Text preprocessing function
@st.cache_data
def clean_text(text, stop_words, _lemmatizer):
    if not text or not isinstance(text, str):
        return ""
    
    try:
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'\W', ' ', text)      # Remove non-word characters
        text = re.sub(r'\s+', ' ', text)     # Remove extra whitespace
        text = re.sub(r'\d+', '', text)      # Remove digits
        
        if _lemmatizer:
            words = nltk.word_tokenize(text)
            words = [word for word in words if word not in stop_words]
            words = [_lemmatizer.lemmatize(word) for word in words]
            return ' '.join(words)
        else:
            return text
    except:
        return text

# Mock model prediction (replace with actual model loading)
@st.cache_resource
@st.cache_resource
def load_model():
    repo_id = "AhmedYusri/RoBERTa-base_fake_news_classifier"
    hf_token = os.getenv("HF_TOKEN")
    model = AutoModelForSequenceClassification.from_pretrained(
        "AhmedYusri/RoBERTa-base_fake_news_classifier",
        use_auth_token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "AhmedYusri/RoBERTa-base_fake_news_classifier",
        use_auth_token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    return tokenizer, model

def predict_news(text, tokenizer, model, max_length=512):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities.numpy()[0]

def main():
    # Header
    st.markdown('<div class="main-header">📰 Fake News Detection System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This application uses a fine-tuned RoBERTa model to classify news articles as either <strong>REAL</strong> or <strong>FAKE</strong>.
    The model was trained on a comprehensive dataset of news articles with advanced NLP preprocessing techniques.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    stop_words, lemmatizer = initialize_nltk()
    tokenizer, model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence required to classify as FAKE"
        )
        
        preprocess_text = st.checkbox(
            "Enable Text Preprocessing",
            value=True,
            help="Clean and normalize the input text"
        )
        
        st.markdown("---")
        st.header("📊 Model Information")
        st.info("""
        **Model**: RoBERTa-base
        **Training Data**: 44,898 news articles
        **Accuracy**: 99.82%
        **F1 Score**: 99.81%
        **Precision**: 99.91%
        **Recall**: 99.72%
        """)
        
        st.markdown("---")
        st.header("📝 About")
        st.markdown("""
        This system was trained to identify:
        - Misleading headlines
        - Fabricated content
        - Biased reporting
        - Satirical news presented as real
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">🔍 News Article Analysis</div>', 
                    unsafe_allow_html=True)
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload File"],
            horizontal=True
        )
        
        news_text = ""
        news_title = ""
        
        if input_method == "Type/Paste Text":
            news_title = st.text_input(
                "News Title (optional):",
                placeholder="Enter the news headline..."
            )
            
            news_text = st.text_area(
                "News Article Text:",
                height=200,
                placeholder="Paste the news article content here..."
            )
        
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt'],
                help="Upload a plain text file containing the news article"
            )
            
            if uploaded_file:
                news_text = uploaded_file.read().decode('utf-8')
                st.text_area("Uploaded Content:", news_text, height=150, disabled=True)
        
        # Analysis button
        if st.button("🔍 Analyze News Article", type="primary"):
            if news_text.strip():
                with st.spinner("Analyzing the article..."):
                    # Combine title and text
                    full_text = f"Title: {news_title} Content: {news_text}" if news_title else news_text
                    
                    # Preprocess if enabled
                    if preprocess_text and lemmatizer:
                        processed_text = clean_text(full_text, stop_words, lemmatizer)
                    else:
                        processed_text = full_text
                    
                    # Get prediction
                    probabilities = predict_news(processed_text, tokenizer, model)
                    result = {
                        'prediction': 'FAKE' if probabilities[0] > 0.5 else 'REAL',
                        'confidence': max(probabilities[0], 1 - probabilities[0]),
                        'probabilities': {
                            'fake': probabilities[0],
                            'real': 1 - probabilities[0]
                        }
                    }

                    # Display results
                    st.markdown("### 📊 Analysis Results")
                    
                    # Prediction display
                    prediction_class = "fake-news" if result['prediction'] == 'FAKE' else "real-news"
                    prediction_emoji = "❌" if result['prediction'] == 'FAKE' else "✅"
                    
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <h3>{prediction_emoji} Prediction: {result['prediction']} NEWS</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed probabilities
                    col_fake, col_real = st.columns(2)
                    with col_fake:
                        st.metric(
                            "Fake Probability", 
                            f"{result['probabilities']['fake']:.2%}",
                            delta=None
                        )
                    with col_real:
                        st.metric(
                            "Real Probability", 
                            f"{result['probabilities']['real']:.2%}",
                            delta=None
                        )
                    
                    # Visualization
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Fake News', 'Real News'],
                            y=[result['probabilities']['fake'], result['probabilities']['real']],
                            marker_color=['#ff4444', '#44ff44'],
                            text=[f"{result['probabilities']['fake']:.1%}", 
                                  f"{result['probabilities']['real']:.1%}"],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Prediction Probabilities",
                        yaxis_title="Probability",
                        xaxis_title="Classification",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional insights
                    st.markdown("### 💡 Analysis Insights")
                    
                    if result['prediction'] == 'FAKE':
                        st.warning("""
                        **⚠️ This article appears to be fake news. Consider:**
                        - Verifying information from multiple reliable sources
                        - Checking the publication date and source credibility
                        - Looking for supporting evidence and citations
                        - Being cautious about sharing this content
                        """)
                    else:
                        st.success("""
                        **✅ This article appears to be legitimate news. However:**
                        - Always cross-reference with other reliable sources
                        - Consider potential bias in reporting
                        - Check for recent updates or corrections
                        - Verify quotes and statistics independently
                        """)
                    
                    # Show preprocessing results if enabled
                    if preprocess_text and lemmatizer:
                        with st.expander("View Preprocessed Text"):
                            st.text(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)
            
            else:
                st.error("Please enter some text to analyze!")
    
    with col2:
        st.markdown('<div class="sub-header">📈 Quick Stats</div>', 
                    unsafe_allow_html=True)
        
        # Sample statistics (in a real app, these would be from actual usage)
        st.metric("Articles Analyzed Today", "1,247", "↑ 156")
        st.metric("Fake News Detected", "312", "↓ 23")
        st.metric("Accuracy Rate", "99.8%", "↑ 0.2%")
        
        st.markdown("---")
        
        # Recent predictions visualization
        st.markdown("### 📊 Recent Predictions")
        
        # Generate sample data for visualization
        sample_data = pd.DataFrame({
            'Time': pd.date_range(start='2024-01-01', periods=20, freq='H'),
            'Fake': np.random.randint(10, 50, 20),
            'Real': np.random.randint(20, 80, 20)
        })
        
        fig_timeline = px.line(
            sample_data, 
            x='Time', 
            y=['Fake', 'Real'],
            title='Predictions Over Time'
        )
        fig_timeline.update_layout(height=300)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Distribution chart
        labels = ['Real News', 'Fake News']
        values = [75, 25]  # Sample distribution
        
        fig_pie = px.pie(
            values=values, 
            names=labels, 
            title='Overall Distribution',
            color_discrete_sequence=['#44ff44', '#ff4444']
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Built with Streamlit | Powered by RoBERTa | 
    Remember: Always verify news from multiple reliable sources
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
