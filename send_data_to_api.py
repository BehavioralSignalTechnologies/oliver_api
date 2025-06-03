#!/usr/bin/env python3
"""
Behavioral Signals API Client

This script provides functionality to send audio files to the 
Behavioral Signals API, process them, and save the results and extracted features.
"""

# Standard library imports
import os
import json
import time
import argparse

# Third-party imports
import requests

# Local imports
import feature_extraction

# Constants
REALTIME_RATIO = 10
BASE_URL = "https://api.behavioralsignals.com/v5"
CONFIG_FILE = "api.config"

# Load configuration
with open(CONFIG_FILE) as f:
    config = json.load(f)
    PROJECT_ID = config["project_id"]
    API_TOKEN = config["api_token"]

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
    
    return response_json, duration, processing_time

def send_audio_and_save_response(file_path):
    """
    Send an audio file to the API, get results, and save them to JSON files.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        tuple: (response_dict, audio_duration, processing_time) or (None, 0, 0) if processing failed
    """
    response, audio_duration, processing_time = send_audio_and_get_response(file_path, os.path.basename(file_path))
    
    if not response:
        print(f"Failed to process {file_path}")
        return None, 0, 0
    
    # Save raw API response
    json_file = file_path.replace(".wav", ".json")
    with open(json_file, "w") as f:
        json.dump(response, f, indent=4)
    print(f"Results saved in: {json_file}")
    
    # Extract and save features
    features_json = feature_extraction.extract_segment_features(response)
    json_features_file = file_path.replace(".wav", "_features.json")
    with open(json_features_file, "w") as f:
        json.dump(features_json, f, indent=4, ensure_ascii=False)
    print(f"Features saved in: {json_features_file}")
    
    return response, audio_duration, processing_time
def main():
    """Main function to parse arguments and process audio files."""
    parser = argparse.ArgumentParser(
        description="Send audio files to Behavioral Signals API for processing"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the audio file or folder with audio files"
    )
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        return
    
    if os.path.isfile(args.input):
        # Process a single file
        if not args.input.endswith(".wav"):
            print("Error: Input file must be a .wav file")
            return
        send_audio_and_save_response(args.input)
    elif os.path.isdir(args.input):
        # Process all .wav files in a directory
        wav_files = [f for f in os.listdir(args.input) if f.endswith(".wav")]
        
        if not wav_files:
            print(f"No .wav files found in {args.input}")
            return
        
        print(f"Found {len(wav_files)} .wav files to process")
        
        # Track total processing time and audio duration
        total_start_time = time.time()
        total_audio_duration = 0.0
        total_processing_time_excluding_upload = 0.0
        
        for file in wav_files:
            file_path = os.path.join(args.input, file)
            print(f"\nProcessing: {file_path}")
            response, audio_duration, processing_time = send_audio_and_save_response(file_path)
            
            # Accumulate totals if processing was successful
            if response:
                total_audio_duration += audio_duration
                total_processing_time_excluding_upload += processing_time
        
        # Calculate total processing time including upload
        total_end_time = time.time()
        total_processing_time_including_upload = total_end_time - total_start_time
        
        # Print comprehensive statistics
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY:")
        print(f"Total files processed: {len(wav_files)}")
        print(f"Total audio duration: {total_audio_duration:.1f} seconds")
        print(f"Total processing time (excluding upload): {total_processing_time_excluding_upload:.1f} seconds")
        print(f"Total processing time (including upload): {total_processing_time_including_upload:.1f} seconds")
        
        if total_audio_duration > 0:
            realtime_ratio_excluding_upload = total_audio_duration / total_processing_time_excluding_upload
            realtime_ratio_including_upload = total_audio_duration / total_processing_time_including_upload
            print(f"Real-time ratio (excluding upload): {realtime_ratio_excluding_upload:.1f}")
            print(f"Real-time ratio (including upload): {realtime_ratio_including_upload:.1f}")
        print(f"{'='*60}")
    else:
        print(f"Error: Invalid input path '{args.input}'")


if __name__ == "__main__":
    main()
