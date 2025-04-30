#!/usr/bin/env python3
"""
Deepfake Evaluation Example

This script evaluates the performance of deepfake detection by:
1. Processing audio files from bonafide and deepfake folders
2. Calling the API to analyze each file
3. Extracting deepfake posteriors from the API response
4. Calculating a confusion matrix based on ground truth labels
"""

# Standard library imports
import os
import argparse
import time
from collections import defaultdict

# Third-party imports
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Local imports
import sys
import os
# Add parent directory to path to import modules from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from send_data_to_api import send_audio_and_get_response
from feature_extraction import extract_segment_features

def process_audio_file(file_path):
    """
    Process a single audio file and determine if it's bonafide or spoofed.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        tuple: (is_spoofed, confidence) where is_spoofed is a boolean and
               confidence is the posterior probability
    """
    print(f"\nProcessing: {file_path}")
    file_name = os.path.basename(file_path)
    
    # Send audio to API and get response
    response = send_audio_and_get_response(file_path, file_name)
    
    if not response:
        print(f"Failed to process {file_path}")
        return None, 0.0
    
    # Extract features from response
    features = extract_segment_features(response)
    
    # Aggregate deepfake posteriors across all segments
    spoofed_posteriors = []
    bonafide_posteriors = []
    
    for segment in features:
        deepfake_posteriors = segment.get("deepfake_posteriors", {})
        
        # Extract posteriors for "spoofed" and "bonafide" labels
        spoofed_posterior = deepfake_posteriors.get("spoofed", 0.0)
        bonafide_posterior = deepfake_posteriors.get("bonafide", 0.0)
        
        spoofed_posteriors.append(spoofed_posterior)
        bonafide_posteriors.append(bonafide_posterior)
    
    # Calculate average posteriors
    avg_spoofed = np.mean(spoofed_posteriors) if spoofed_posteriors else 0.0
    avg_bonafide = np.mean(bonafide_posteriors) if bonafide_posteriors else 0.0
    
    # Determine if the file is classified as spoofed
    is_spoofed = avg_spoofed > avg_bonafide
    confidence = max(avg_spoofed, avg_bonafide)
    
    print(f"Classification: {'Spoofed' if is_spoofed else 'Bonafide'} (confidence: {confidence:.4f})")
    return is_spoofed, confidence

def evaluate_folder(folder_path):
    """
    Evaluate all audio files in the bonafide and deepfake subfolders.
    
    Args:
        folder_path (str): Path to the folder containing bonafide and deepfake subfolders
        
    Returns:
        tuple: (y_true, y_pred) arrays for confusion matrix calculation
    """
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return [], []
    
    # Check if bonafide and deepfake subfolders exist
    bonafide_folder = os.path.join(folder_path, "bonafide")
    deepfake_folder = os.path.join(folder_path, "deepfake")
    
    if not os.path.isdir(bonafide_folder):
        print(f"Error: Bonafide folder '{bonafide_folder}' does not exist")
        return [], []
    
    if not os.path.isdir(deepfake_folder):
        print(f"Error: Deepfake folder '{deepfake_folder}' does not exist")
        return [], []
    
    # Get all WAV files in both folders
    bonafide_files = [os.path.join(bonafide_folder, f) for f in os.listdir(bonafide_folder) if f.endswith(".wav")]
    deepfake_files = [os.path.join(deepfake_folder, f) for f in os.listdir(deepfake_folder) if f.endswith(".wav")]
    
    print(f"Found {len(bonafide_files)} bonafide files and {len(deepfake_files)} deepfake files")
    
    # Initialize arrays for ground truth and predictions
    y_true = []
    y_pred = []
    confidences = []
    file_results = []
    
    # Process bonafide files
    print("\nProcessing bonafide files...")
    for file_path in bonafide_files:
        result, confidence = process_audio_file(file_path)
        if result is not None:  # Only include if processing was successful
            y_true.append(0)  # 0 for bonafide (ground truth)
            y_pred.append(1 if result else 0)  # 1 for spoofed, 0 for bonafide (prediction)
            confidences.append(confidence)
            file_results.append((os.path.basename(file_path), "bonafide", "spoofed" if result else "bonafide", confidence))
    
    # Process deepfake files
    print("\nProcessing deepfake files...")
    for file_path in deepfake_files:
        result, confidence = process_audio_file(file_path)
        if result is not None:  # Only include if processing was successful
            y_true.append(1)  # 1 for spoofed (ground truth)
            y_pred.append(1 if result else 0)  # 1 for spoofed, 0 for bonafide (prediction)
            confidences.append(confidence)
            file_results.append((os.path.basename(file_path), "deepfake", "spoofed" if result else "bonafide", confidence))
    
    # Print detailed results
    print("\nDetailed Results:")
    print("=" * 80)
    print(f"{'Filename':<30} {'Ground Truth':<15} {'Prediction':<15} {'Confidence':<10}")
    print("-" * 80)
    for filename, truth, pred, conf in file_results:
        print(f"{filename:<30} {truth:<15} {pred:<15} {conf:.4f}")
    print("=" * 80)
    
    return np.array(y_true), np.array(y_pred), np.array(confidences)

def main():
    """Main function to parse arguments and evaluate deepfake detection."""
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection performance on a folder of audio files"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the folder containing bonafide and deepfake subfolders"
    )
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        return
    
    print(f"Evaluating deepfake detection on folder: {args.input}")
    start_time = time.time()
    
    # Evaluate the folder
    y_true, y_pred, confidences = evaluate_folder(args.input)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No results to evaluate")
        return
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("=" * 40)
    print(f"{'':>10} {'Predicted':>20}")
    print(f"{'':>10} {'-'*30}")
    print(f"{'':>10} {'Bonafide':>10} {'Spoofed':>10}")
    print(f"{'Actual':>10} {'-'*30}")
    print(f"{'Bonafide':>10} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"{'Spoofed':>10} {cm[1][0]:>10} {cm[1][1]:>10}")
    print("=" * 40)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print metrics
    print("\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Bonafide", "Spoofed"]))
    
    # Print execution time
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
