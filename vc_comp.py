import numpy as np
import librosa
import soundfile as sf
from pesq import pesq
from pesq import PesqError
from pystoi import stoi
import scipy.signal as sps
from scipy.linalg import norm
import argparse

def load_wav(file_path, target_sr=None):
    wav, sr = librosa.load(file_path, sr=target_sr)
    return wav, sr

def resample_if_necessary(wav, original_sr, target_sr=16000):
    if original_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=original_sr, target_sr=target_sr)
    return wav

def trim_to_shortest(reference, test):
    min_len = min(len(reference), len(test))
    return reference[:min_len], test[:min_len]

def compute_mae(reference, test):
    mae = np.mean(np.abs(reference - test))
    return mae

def compute_mse(reference, test):
    mse = np.mean((reference - test) ** 2)
    return mse

def compute_snr(reference, test):
    noise = reference - test
    snr = 10 * np.log10(np.sum(reference ** 2) / np.sum(noise ** 2))
    return snr

def compute_pesq(reference, sr_ref, test, sr_test):
    if sr_ref != sr_test:
        raise ValueError("Sample rates of reference and test signals must be the same for PESQ.")
    try:
        pesq_score = pesq(sr_ref, reference, test, 'wb')
    except PesqError as e:
        pesq_score = None
    return pesq_score

def compute_stoi(reference, sr_ref, test, sr_test):
    if sr_ref != sr_test:
        raise ValueError("Sample rates of reference and test signals must be the same for STOI.")
    stoi_score = stoi(reference, test, sr_ref, extended=False)
    return stoi_score

def compute_llr(reference, test, sr):
    ref_cepstrum = librosa.feature.mfcc(y=reference, sr=sr)
    test_cepstrum = librosa.feature.mfcc(y=test, sr=sr)
    llr = np.mean(np.sum((ref_cepstrum - test_cepstrum) ** 2, axis=0))
    return llr

def compute_cepstral_distance(reference, test, sr):
    ref_cepstrum = librosa.feature.mfcc(y=reference, sr=sr)
    test_cepstrum = librosa.feature.mfcc(y=test, sr=sr)
    cd = np.mean(np.sqrt(np.sum((ref_cepstrum - test_cepstrum) ** 2, axis=0)))
    return cd

def compute_isd(reference, test):
    ref_spectrum = np.abs(librosa.stft(reference))
    test_spectrum = np.abs(librosa.stft(test))
    isd = np.mean((ref_spectrum - test_spectrum) ** 2 / (ref_spectrum * test_spectrum))
    return isd

def evaluate_model(reference_paths, test_paths):
    target_sr = 16000
    metrics = {
        'MAE': [],
        'MSE': [],
        'SNR': [],
        'PESQ': [],
        'STOI': [],
        'LLR': [],
        'CD': [],
        'ISD': []
    }

    for ref_path, test_path in zip(reference_paths, test_paths):
        reference, sr_ref = load_wav(ref_path, target_sr=target_sr)
        test, sr_test = load_wav(test_path, target_sr=target_sr)
        
        reference_trimmed, test_trimmed = trim_to_shortest(reference, test)
        
        metrics['MAE'].append(compute_mae(reference_trimmed, test_trimmed))
        metrics['MSE'].append(compute_mse(reference_trimmed, test_trimmed))
        metrics['SNR'].append(compute_snr(reference_trimmed, test_trimmed))
        metrics['PESQ'].append(compute_pesq(reference_trimmed, sr_ref, test_trimmed, sr_test))
        metrics['STOI'].append(compute_stoi(reference_trimmed, sr_ref, test_trimmed, sr_test))
        metrics['LLR'].append(compute_llr(reference_trimmed, test_trimmed, sr_ref))
        metrics['CD'].append(compute_cepstral_distance(reference_trimmed, test_trimmed, sr_ref))
        metrics['ISD'].append(compute_isd(reference_trimmed, test_trimmed))
    
    average_metrics = {key: np.nanmean(value) for key, value in metrics.items()}
    return average_metrics

def main(reference_paths, test_paths):
    model_names = set(path.split('_')[0] for path in test_paths)
    results = {}
    for model in model_names:
        model_test_paths = [path for path in test_paths if path.startswith(model)]
        results[model] = evaluate_model(reference_paths, model_test_paths)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the quality of voice cloning models.')
    parser.add_argument('reference', type=str, nargs=5, help='Paths to the reference .wav files')
    parser.add_argument('test', type=str, nargs='+', help='Paths to the test .wav files')

    args = parser.parse_args()
    reference_paths = args.reference
    test_paths = args.test

    results = main(reference_paths, test_paths)
    for model, result in results.items():
        print(f"Results for model {model}:")
        for metric, value in result.items():
            print(f"{metric}: {value}")
        print()
