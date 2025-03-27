# Oliver API Python Usage Example

## 📌 Introduction

This repo demonstrates how to use the [Oliver API](https://oliver.behavioralsignals.com) to send speech data and retrieve predictions related to emotions and behaviors using Python code. [Oliver API](https://oliver.behavioralsignals.com) offers predictions related to:
 - speech emotions and behaviors 
 - speech2text transcriptions
 - speech deep fake detections (i.e. if an utterance is a deep fake or bonafide audio)

## ⚙️ Setup Instructions

### Get API Credentials

- Visit [Oliver API](https://oliver.behavioralsignals.com) and sign up
- Create a project to obtain  `project_id` and `api_token` (See [here](https://oliver.readme.io/docs/create-a-project-and-api-token) for particular instructions)

Then, create a config file named `oliver_api.config` in the following format:

```json
{
  "project_id": "1002141210",
  "api_token": "oar32932fj0isjf023j01f329j"
}
```

and place it with that name in the folder of this repo.

### Set Up Environment

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

---

## 🏢 About Behavioral Signals

Behavioral Signals is a company at the forefront of Emotion AI, building technology that enables machines to understand human behavior and emotions from voice. Our mission is to bridge the gap between humans and machines by leveraging advanced speech analysis and behavioral prediction technologies. Learn more at [behavioralsignals.com](https://www.behavioralsignals.com).

![Behavioral Signals Logo](BehavioralSignals_logo_900px.png)
