# Indian Sign Language Detection and Recognition

Real-time Indian Sign Language (ISL) recognition using **Computer Vision (OpenCV + MediaPipe)** and **Deep Learning (TensorFlow/Keras)**. The webcam captures a hand gesture, the system recognizes the sign, shows it as text and can speak it aloud.

> B.Tech (Computer Science & Engineering, specialization in Artificial Intelligence) final-year project
> School of Engineering, Babu Banarasi Das University, Lucknow


# Problem
People who use sign language face communication barriers with those who do not understand it, and human interpreters are not always available. This project provides an affordable, webcam-only tool that translates selected ISL gestures into text and speech.

# Features
- Real-time webcam hand detection (MediaPipe, 21 landmarks)
- ISL alphabet, number and selected common-word recognition
- Landmark normalization (independent of hand position and distance)
- Keras neural network + RandomForest baseline for comparison
- Confidence score, threshold and temporal smoothing
- Sentence building from recognized signs
- Text-to-speech output
- SQLite recognition history
- Streamlit web interface and OpenCV live window
- Automated tests and CI

# Architecture

Webcam -> Hand Detection (MediaPipe) -> Normalization -> Feature Extraction (83 features)
       -> Keras Classifier -> Smoothing -> Text | Speech | SQLite History

# Methodology
1. **Data collection** - labelled landmark samples with variation in users, lighting, distance, orientation
2. **Preprocessing** - wrist-relative, scale-normalized landmarks
3. **Features** - 63 coordinates + 10 fingertip distances + 10 joint angles
4. **Model** - MLP (256-128-64, BatchNorm, Dropout) with EarlyStopping; RandomForest baseline
5. **Split** - 70% train / 15% validation / 15% test (stratified)
6. **Metrics** - accuracy, precision, recall, F1, loss, confusion matrix
7. **Real-time** - majority-vote smoothing and hold-time to build text

# Results
> Fill in **after** training on your dataset. Do not report numbers you have not measured. See [docs/TESTING.md](docs/TESTING.md).

| Metric | Keras MLP | RandomForest |
|---|---|---|
| Accuracy | – | – |
| F1 (macro) | – | – |

Add `reports/confusion_matrix_keras.png`, `reports/training_curves.png` and a demo GIF/screenshot here.

# Project structure
```
├── app.py                 Streamlit UI
├── realtime.py            OpenCV live recognition
├── collect_data.py        Dataset collection
├── train_model.py         Training + reports
├── evaluate.py            Test evaluation + speed
├── config.py              Paths and settings
├── src/                   hand_detector, features, dataset, predictor, database, tts
├── tests/                 pytest unit tests
├── docs/                  Architecture, testing, dataset notes
├── data/  models/  reports/
└── .github/workflows/     CI
```

# Limitations
- Recognizes static signs only (no continuous sentences yet)
- Single hand
- Accuracy depends on dataset size, number of classes and camera conditions

# Future work
Dynamic gestures (LSTM/Transformer over landmark sequences), two-hand signs, continuous sign-language translation, mobile app, multilingual speech output.

# Team
| Name | Enrollment No. |
|---|---|
| Krishna Jaiswal | 1230439162 |
| Mohd Adeeb Ali | 1230439176 |
| Mohd Owais Khan | 1230439185 |
| Kartik Singh | 1230439157 |

**Supervisor:** Ms. Surabhi Mishra, Department of Computer Science & Engineering, BBD University, Lucknow


## Acknowledgements
MediaPipe, OpenCV, TensorFlow/Keras, scikit-learn, Streamlit.
