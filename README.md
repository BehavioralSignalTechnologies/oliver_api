# Oliver API Python Usage Example

## 📌 Introduction

This small repo demonstrates how to use the [Oliver API](https://oliver.behavioralsignals.com) to send speech data and retrieve predictions related to emotions and behaviors.

## ⚙️ Setup Instructions

### 1. Get API Credentials

- Visit [Oliver API](https://oliver.behavioralsignals.com)
- Create a project to obtain:
  - `project_id`
  - `api_token`

Create a config file named `oliver_api.config` in the following format:

```json
{
  "project_id": "1002141210",
  "api_token": "oar32932fj0isjf023j01f329j"
}
```

### 2. Set Up Environment

```bash
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## 🚀 Usage

Run the following command:

```bash
python3 send_data_to_api.py -i PATH_TO_AUDIO_FILE
```

If the script runs successfully, two JSON files will be generated:

1. `PATH_TO_AUDIO_FILE.json`  
   Contains the initial response from the OLIVER API.

2. `PATH_TO_AUDIO_FILE_features.json`  
   Contains an "aggregated" output with predictions per speech utterance.

### 🔍 Example Output

Example `*_features.json` (for a single utterance):

```json
[
  {
    "id": "0",
    "startTime": "0.411",
    "endTime": "4.351",
    "speaker_ID": "SPEAKER_00",
    "language": "en",
    "language_posterior": 0.79541015625,
    "transcription": "he settled for the remainder of his life in Athens",
    "gender": "male",
    "gender_posterior": 1.0,
    "age_estimate": 54.534549999999996,
    "emotion_posteriors": {
      "happy": 0.0763,
      "sad": 0.0319,
      "angry": 0.0215
    },
    "positivity_posteriors": {
      "positive": 0.1202,
      "negative": 0.0619
    },
    "strength_posteriors": {
      "weak": 0.4849,
      "strong": 0.0053
    },
    "speaking_rate": -0.276,
    "hesitation_posterior": 0.1001,
    "deepfake_posteriors": {
      "spoofed": 1.0,
      "bonafide": 0.0
    }
  }
]
```
