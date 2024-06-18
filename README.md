# Voice cloning models comparison
This project aims to compare the quality of various voice cloning models using following metrics:
- SNR (Signal-to-Noise Ratio)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- LLR (Log Likelihood Ratio)
- CD (Cepstral Distance)
- ISD (Itakura-Saito Distance)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)

# Usage
```vc_comp.py [-h] reference test [test ...]```

- ```reference```   Path to the reference .wav file
- ```test```        Paths to the test .wav files
