#!/usr/bin/env python3
"""
Streamlit Frontend for Audio Analysis Summarization

This frontend provides a user interface for uploading audio files,
processing them through the Behavioral Signals API, and displaying
ChatGPT-generated summaries.
"""

import streamlit as st
import requests
import json
import time
from io import BytesIO

# Configure Streamlit page
st.set_page_config(
    page_title="Audio Analysis Summarization",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL
BACKEND_URL = "http://localhost:8000"

def check_backend_health():
    """Check if the backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_and_analyze_audio(audio_file):
    """
    Upload audio file to backend and get analysis summary.
    
    Args:
        audio_file: Streamlit uploaded file object
        
    Returns:
        dict: Response from backend or None if failed
    """
    try:
        # Prepare file for upload
        files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
        
        # Send request to backend
        response = requests.post(
            f"{BACKEND_URL}/analyze-audio",
            files=files,
            timeout=300  # 5 minutes timeout for processing
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. The audio file might be too long or the server is busy.")
        return None
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def display_analysis_results(results):
    """
    Display the analysis results in the UI.
    
    Args:
        results (dict): Analysis results from backend
    """
    # Main summary
    st.subheader("📝 Analysis Summary")
    st.write(results["summary"])
    
    # Audio information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Audio Duration", f"{results['audio_duration']:.1f}s")
    
    with col2:
        st.metric("Processing Time", f"{results['processing_time']:.1f}s")
    
    with col3:
        if results['audio_duration'] > 0:
            ratio = results['audio_duration'] / results['processing_time']
            st.metric("Real-time Ratio", f"{ratio:.1f}x")
    
    # Raw analysis data (expandable)
    with st.expander("🔍 View Raw Analysis Data"):
        st.json(results["raw_analysis"])

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🎵 Audio Analysis Summarization")
    st.markdown("Upload an audio file to get AI-powered analysis and insights about speech patterns, emotions, and characteristics.")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        **Supported formats:**
        - WAV files
        - MP3 files
        
        **Analysis includes:**
        - Speech transcription
        - Emotion detection
        - Speaking rate analysis
        - Gender and age estimation
        - Engagement levels
        - Sentiment analysis
        """)
        
        st.header("🔧 Backend Status")
        if check_backend_health():
            st.success("✅ Backend is running")
        else:
            st.error("❌ Backend is not accessible")
            st.markdown("Make sure to start the backend server:")
            st.code("python backend.py")
    
    # Main content area
    st.header("📁 Upload Audio File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3'],
        help="Upload a WAV or MP3 file for analysis"
    )
    
    if uploaded_file is not None:
        # Display file information
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write("**File details:**")
            st.write(f"- Name: {uploaded_file.name}")
            st.write(f"- Size: {uploaded_file.size / 1024 / 1024:.2f} MB")
            st.write(f"- Type: {uploaded_file.type}")
        
        with col2:
            # Audio player
            st.write("**Preview:**")
            st.audio(uploaded_file.getvalue())
        
        # Analysis button
        st.header("🚀 Start Analysis")
        
        if st.button("Analyze Audio", type="primary", use_container_width=True):
            # Check backend health before processing
            if not check_backend_health():
                st.error("❌ Backend server is not running. Please start the backend first.")
                st.stop()
            
            # Show progress
            with st.spinner("🔄 Processing audio file..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress indicators
                status_text.text("Uploading file...")
                progress_bar.progress(20)
                
                # Process the file
                results = upload_and_analyze_audio(uploaded_file)
                
                if results:
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Display results
                    st.success("🎉 Analysis completed successfully!")
                    display_analysis_results(results)
                    
                else:
                    progress_bar.progress(0)
                    status_text.text("❌ Analysis failed")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an audio file to begin analysis.")
        
        # Example section
        st.header("📖 How it works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Upload**
            
            Upload your WAV or MP3 audio file using the file uploader above.
            """)
        
        with col2:
            st.markdown("""
            **2. Analyze**
            
            The audio is processed using advanced AI models to extract speech patterns, emotions, and characteristics.
            """)
        
        with col3:
            st.markdown("""
            **3. Summarize**
            
            ChatGPT analyzes the results and provides a comprehensive, easy-to-understand summary.
            """)

if __name__ == "__main__":
    main()
