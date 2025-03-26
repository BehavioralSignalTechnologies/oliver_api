# per segment feature extraction function:
def extract_segment_features(data):
    # Define age centers for linear combination
    age_centers = {
        "18 - 22": 20,
        "23 - 30": 26.5,
        "31 - 45": 38,
        "46 - 65": 55.5
    }

    # Initialize the list for processed segments
    segments = []

    # Group results by segment ID
    segment_dict = {}
    for result in data["results"]:
        segment_id = result["id"]
        if segment_id not in segment_dict:
            segment_dict[segment_id] = []
        segment_dict[segment_id].append(result)

    # Process each segment
    for segment_id, segment_results in segment_dict.items():
        segment_data = {
            "id": segment_id,
            "startTime": None,
            "endTime": None,
            "speaker_ID": None,
            "language": None,
            "language_posterior": None,
            "transcription": None,
            "gender": None,
            "gender_posterior": None,
            "age_estimate": None,
            "emotion_posteriors": {},
            "positivity_posteriors": {},
            "strength_posteriors": {},
            "speaking_rate": 0,
            "hesitation_posterior": None,
            "deepfake_posteriors": {}
        }

        for result in segment_results:
            # Extract start and end times
            segment_data["startTime"] = result["startTime"]
            segment_data["endTime"] = result["endTime"]

            # Process each task
            task = result["task"]
            if task == "diarization":
                segment_data["speaker_ID"] = result["finalLabel"]
            elif task == "language":
                top_language = max(result["prediction"], key=lambda x: float(x["posterior"]))
                segment_data["language"] = top_language["label"]
                segment_data["language_posterior"] = float(top_language["posterior"])
            elif task == "asr":
                segment_data["transcription"] = result["finalLabel"].strip()
            elif task == "gender":
                top_gender = max(result["prediction"], key=lambda x: float(x["posterior"]))
                segment_data["gender"] = top_gender["label"]
                segment_data["gender_posterior"] = float(top_gender["posterior"])
            elif task == "age":
                # Compute weighted age estimate
                segment_data["age_estimate"] = sum(
                    age_centers[p["label"]] * float(p["posterior"])
                    for p in result["prediction"]
                    if p["label"] in age_centers
                )
            elif task == "emotion":
                # Exclude neutral emotions
                segment_data["emotion_posteriors"] = {
                    p["label"]: float(p["posterior"])
                    for p in result["prediction"]
                    if p["label"] != "neutral"
                }
            elif task == "positivity":
                # Exclude neutral positivity
                segment_data["positivity_posteriors"] = {
                    p["label"]: float(p["posterior"])
                    for p in result["prediction"]
                    if p["label"] != "neutral"
                }
            elif task == "strength":
                # Exclude neutral strengths
                segment_data["strength_posteriors"] = {
                    p["label"]: float(p["posterior"])
                    for p in result["prediction"]
                    if p["label"] != "neutral"
                }
            elif task == "speaking_rate":
                # Calculate speaking rate
                slow_rate = sum(float(p["posterior"]) for p in result["prediction"] if p["label"] in ["slow", "very slow"])
                fast_rate = sum(float(p["posterior"]) for p in result["prediction"] if p["label"] in ["fast", "very fast"])
                segment_data["speaking_rate"] = fast_rate - slow_rate
            elif task == "hesitation":
                # Extract hesitation "yes" posterior
                hesitation_yes = next((p for p in result["prediction"] if p["label"] == "yes"), None)
                if hesitation_yes:
                    segment_data["hesitation_posterior"] = float(hesitation_yes["posterior"])
            elif task == "deepfake":
                # Extract deepfake posteriors
                segment_data["deepfake_posteriors"] = {
                    p["label"]: float(p["posterior"])
                    for p in result["prediction"]
                }

        # Append processed segment
        segments.append(segment_data)

    return segments
