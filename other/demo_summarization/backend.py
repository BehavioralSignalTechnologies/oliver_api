#!/usr/bin/env python3
"""
FastAPI Backend for Audio Analysis Summarization

This backend handles audio file uploads, processes them using the Behavioral Signals API,
and generates summaries using ChatGPT.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
from pydub import AudioSegment

# Import the API client functionality
from api_client import send_audio_and_get_response, extract_segment_features

app = FastAPI(title="Audio Analysis Summarization API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    """Load configuration from config.json"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")

def convert_to_wav(input_file_path: str, output_file_path: str) -> bool:
    """
    Convert audio file to WAV format using pydub.
    
    Args:
        input_file_path (str): Path to input audio file
        output_file_path (str): Path for output WAV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_file_path)
        
        # Export as WAV
        audio.export(output_file_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False

def format_features_for_gpt(segments: list) -> str:
    """
    Format the processed segment features for ChatGPT analysis.
    
    Args:
        segments (list): List of processed segment features
        
    Returns:
        str: Formatted text for ChatGPT
    """
    if not segments:
        return "No analysis results available."
    
    formatted_text = "Audio Analysis Summary:\n\n"
    
    for i, segment in enumerate(segments, 1):
        formatted_text += f"Segment {i} ({segment['startTime']}s - {segment['endTime']}s):\n"
        
        # Transcription
        if segment.get('transcription'):
            formatted_text += f"  Transcript: {segment['transcription']}\n"
        
        # Speaker and language info
        if segment.get('speaker_ID'):
            formatted_text += f"  Speaker: {segment['speaker_ID']}\n"
        if segment.get('language'):
            formatted_text += f"  Language: {segment['language']} (confidence: {segment.get('language_posterior', 'N/A'):.2f})\n"
        
        # Demographics
        if segment.get('gender'):
            formatted_text += f"  Gender: {segment['gender']} (confidence: {segment.get('gender_posterior', 'N/A'):.2f})\n"
        if segment.get('age_estimate'):
            formatted_text += f"  Estimated Age: {segment['age_estimate']:.1f} years\n"
        
        # Emotional characteristics
        if segment.get('emotion_posteriors'):
            emotions = ", ".join([f"{emotion}: {score:.2f}" for emotion, score in segment['emotion_posteriors'].items()])
            formatted_text += f"  Emotions: {emotions}\n"
        
        if segment.get('positivity_posteriors'):
            positivity = ", ".join([f"{pos}: {score:.2f}" for pos, score in segment['positivity_posteriors'].items()])
            formatted_text += f"  Sentiment: {positivity}\n"
        
        if segment.get('strength_posteriors'):
            strength = ", ".join([f"{str_type}: {score:.2f}" for str_type, score in segment['strength_posteriors'].items()])
            formatted_text += f"  Voice Strength: {strength}\n"
        
        # Speaking characteristics
        if segment.get('speaking_rate') is not None:
            rate_desc = "fast" if segment['speaking_rate'] > 0.1 else "slow" if segment['speaking_rate'] < -0.1 else "normal"
            formatted_text += f"  Speaking Rate: {rate_desc} (score: {segment['speaking_rate']:.2f})\n"
        
        if segment.get('hesitation_posterior'):
            formatted_text += f"  Hesitation: {segment['hesitation_posterior']:.2f}\n"
        
        # Authenticity
        if segment.get('deepfake_posteriors'):
            deepfake = ", ".join([f"{auth}: {score:.2f}" for auth, score in segment['deepfake_posteriors'].items()])
            formatted_text += f"  Authenticity: {deepfake}\n"
        
        formatted_text += "\n"
    
    return formatted_text

async def get_chatgpt_summary(analysis_text: str, config: dict) -> str:
    """
    Get summary from ChatGPT API.
    
    Args:
        analysis_text (str): Formatted analysis text
        config (dict): Configuration containing API key and model
        
    Returns:
        str: ChatGPT summary
    """
    try:
        # Set up OpenAI client
        openai.api_key = config['openai_api_key']
        
        # Create the prompt
        prompt = f"""Please analyze the following audio analysis results and provide a comprehensive summary. 
Focus on the key insights about the speaker's emotional state, speaking patterns, and overall characteristics:

{analysis_text}

Please provide a clear, concise summary that highlights:
1. The main emotional tone and sentiment
2. Turn taking, speaker attributes and interaction dynamics: use the durations of each utterance of each speaker to assess the turn taking and interaction dynamics. Who speaks more, who interrupts, how long does each speaker speak, speaking rate, etc.
3. Overall asseessment of the subject under discussion and the main topics discussed

Summary:"""

        # Call ChatGPT API
        response = openai.ChatCompletion.create(
            model=config.get('model', 'gpt-3.5-turbo'),
            messages=[
                {"role": "system", "content": "You are an expert audio analyst who provides clear, insightful summaries of speech analysis data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChatGPT API error: {str(e)}")

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file and return ChatGPT summary.
    
    Args:
        file: Uploaded audio file (WAV or MP3)
        
    Returns:
        JSON response with analysis summary
    """
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only WAV and MP3 files are supported")
    
    # Load configuration
    config = load_config()
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded file
            input_file_path = os.path.join(temp_dir, file.filename)
            with open(input_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Convert to WAV if necessary
            if file.filename.lower().endswith('.mp3'):
                wav_file_path = os.path.join(temp_dir, "converted.wav")
                if not convert_to_wav(input_file_path, wav_file_path):
                    raise HTTPException(status_code=500, detail="Failed to convert MP3 to WAV")
            else:
                wav_file_path = input_file_path
            
            # Process with Behavioral Signals API
            response_data, audio_duration, processing_time = send_audio_and_get_response(
                wav_file_path, 
                os.path.basename(file.filename)
            )
            
            if not response_data:
                raise HTTPException(status_code=500, detail="Failed to process audio with Behavioral Signals API")
            
            # Extract processed features
            segment_features = extract_segment_features(response_data)
            
            # Format features for ChatGPT
            formatted_analysis = format_features_for_gpt(segment_features)
            
            # Get ChatGPT summary
            summary = await get_chatgpt_summary(formatted_analysis, config)
            
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "summary": summary,
                "raw_analysis": response_data
            })
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
