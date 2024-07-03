# Voice cloning models comparison
This project aims to compare the quality of various voice cloning models using the following metrics:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- SNR (Signal-to-Noise Ratio)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- LLR (Log Likelihood Ratio)
- CD (Cepstral Distance)
- ISD (Itakura-Saito Distance)

# Usage
```vc_comp.py [-h] reference test [test ...]```

```reference``` -path to the reference .wav file
```test``` -paths to the test .wav files

# Example
## Comparison of Tacotron and FastSpeech voice cloning models based on two WAV samples:  
```vc_comp.py reference_1.wav reference_2.wav tacotron_1.wav tacotron_2.wav fastspeech_1.wav fastspeech_2.wav```

