#!/usr/bin/env python3
"""
API Client for Behavioral Signals API

This module contains the necessary functions from send_data_to_api.py
to work with the Behavioral Signals API.
"""

import os
import json
import time
import requests
import sys

# Add root directory to path to import feature_extraction
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)

# Import feature extraction function
from feature_extraction import extract_segment_features

# Load configuration from the root directory
def load_api_config():
    """Load API configuration from the root directory"""
    # Get the root directory (two levels up from this file)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_file = os.path.join(root_dir, 'api.config')
    
    try:
        with open(config_file) as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise Exception(f"Failed to load API config from {config_file}: {e}")

# Load configuration
config = load_api_config()
PROJECT_ID = config["project_id"]
API_TOKEN = config["api_token"]

# Constants
REALTIME_RATIO = 10
BASE_URL = "https://api.behavioralsignals.com/v5"

# API endpoints
UPLOAD_URL = f"{BASE_URL}/clients/{PROJECT_ID}/processes/audio"

# Request headers
HEADERS = {
    "X-Auth-Token": API_TOKEN
}

def send_audio_file(file_path, file_name):
    """
    Upload an audio file to the API.
    
    Args:
        file_path (str): Path to the audio file
        file_name (str): Name to use for the uploaded file
        
    Returns:
        dict: JSON response from the API or None if upload failed
    """
    try:
        with open(file_path, "rb") as audio_file:
            files = {
                "file": audio_file,
                "name": (None, file_name)
            }
            response = requests.post(UPLOAD_URL, headers=HEADERS, files=files)
        
        response.raise_for_status()
        print("Upload successful!")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to upload: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            print(f"Error message: {e.response.text}")
        return None

def check_process(process_id, client_id):
    """
    Check the status of a processing job.
    
    Args:
        process_id (str): Process ID to check
        client_id (str): Client ID
        
    Returns:
        dict: JSON response with process status
    """
    process_url = f"{BASE_URL}/clients/{client_id}/processes/{process_id}"
    response = requests.get(process_url, headers=HEADERS)
    return response.json()

def send_audio_and_get_response(file_path, file_name):
    """
    Send an audio file to the API, monitor processing, and get results.
    
    Args:
        file_path (str): Path to the audio file
        file_name (str): Name to use for the uploaded file
        
    Returns:
        tuple: (response_dict, audio_duration, processing_time) or (None, 0, 0) if processing failed
    """
    print("Sending audio file to API...")
    upload_response = send_audio_file(file_path, file_name)
    
    if not upload_response:
        return None, 0, 0
    
    print("Processing audio file:")
    start_time = time.time()
    
    while True:
        process_response = check_process(upload_response["pid"], PROJECT_ID)
        
        # Status 2 means processing is complete
        if process_response["status"] == 2:
            break
        # Status 1 means processing is in progress
        elif process_response["status"] == 1:
            current_time = time.time()
            duration = process_response["duration"]
            elapsed = current_time - start_time
            
            # Calculate progress percentage
            percentage_processed = min(1.0, elapsed * REALTIME_RATIO / duration)
            print(f"Please wait... {100 * percentage_processed:.1f}% completed", end="\r")
        elif process_response["status"] == 0: # API busy with another job:
            print("API is busy, waiting...")
            
        time.sleep(0.5)
    
    end_time = time.time()
    processing_time = end_time - start_time
    duration = process_response["duration"]
    
    print("DONE                                              ")
    print(f"Audio duration: {duration:.1f} seconds")
    print(f"Processing took {processing_time:.1f} seconds")
    
    realtime_ratio = duration / processing_time
    print(f"Real-time ratio: {realtime_ratio:.1f}")

    # Get the results
    results_url = f"{BASE_URL}/clients/{PROJECT_ID}/processes/{process_response['pid']}/results"
    results_response = requests.get(results_url, headers=HEADERS)
    
    response_json = results_response.json()
    
    # Extract audio duration from results (find maximum endTime)
    audio_duration = 0.0
    if 'results' in response_json and response_json['results']:
        for result in response_json['results']:
            if 'endTime' in result:
                end_time_val = float(result['endTime'])
                audio_duration = max(audio_duration, end_time_val)
    
    return response_json, audio_duration, processing_time
