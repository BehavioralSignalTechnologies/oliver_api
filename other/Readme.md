# Deepfake Evaluation Script

This folder contains the `df_evaluation_example.py` script, designed for evaluating the **Speaker-Agnostic Deep Fake Detection** feature of the [Oliver API](http://oliver.behavioralsignals.com). The script performs batch inference on labeled audio data and reports standard classification metrics.

## 🔍 Script Overview

`df_evaluation_example.py`:
- Sends labeled audio data (bonafide & spoofed) to the Oliver API
- Collects results from the API
- Computes performance metrics including:
  - Confusion Matrix
  - Accuracy, Precision, Recall, F1 Score
  - Full classification report

---

## 📁 Expected Directory Structure

Before running the script, prepare your input directory like this:

```
sample_data/speaker_agnostic/
├── bonafide/
│   ├── real_audio1.wav
│   ├── real_audio2.wav
│   └── ...
└── deepfake/
    ├── fake_audio1.wav
    ├── fake_audio2.wav
    └── ...
```

- The **`bonafide`** subfolder should contain genuine human speech.
- The **`deepfake`** subfolder should contain synthetic or spoofed speech.

---

## ▶️ Usage

After setting up your Python environment and installing dependencies:

```bash
python3 df_evaluation_example.py -i sample_data/speaker_agnostic
```

---

## 📊 Example Output

```
Confusion Matrix:
========================================
                      Predicted
           ------------------------------
             Bonafide    Spoofed
    Actual ------------------------------
  Bonafide         18          2
   Spoofed          4         16
========================================

Metrics:
Accuracy:  0.8500
Precision: 0.8889
Recall:    0.8000
F1 Score:  0.8421
```

---

## ⚠️ Notes

- This script assumes you've already signed up for the [Oliver API](http://oliver.behavioralsignals.com) and have access to a valid **API key**.
- You must also have the `send_data_to_api.py` dependencies installed and configured.

---

## 📫 Contact

For questions or support, please refer to the main [Oliver API documentation](http://oliver.behavioralsignals.com) or contact your Behavioral Signals representative.
