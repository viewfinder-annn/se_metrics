import numpy as np
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
import soundfile as sf
import torch
import torchaudio
import argparse
import json
from tqdm import tqdm

def evaluate_from_audio(noisy_dir, clean_dir, saved_dir, target_sr=None):
    metrics_total = np.zeros(6)
    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    for audio in tqdm(audio_list):
        saved_path = os.path.join(saved_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        est_audio, est_sr = torchaudio.load(saved_path)
        clean_audio, clean_sr = torchaudio.load(clean_path)
        # Resample to target_sr
        if target_sr is not None:
            est_audio = torchaudio.transforms.Resample(est_sr, target_sr)(est_audio)
            clean_audio = torchaudio.transforms.Resample(clean_sr, target_sr)(clean_audio)
            est_audio = est_audio.squeeze().numpy()
            clean_audio = clean_audio.squeeze().numpy()
            # est_audio = est_audio[:len(clean_audio)]
        else:
            target_sr = sr
        try:
            metrics = compute_metrics(clean_audio, est_audio, target_sr, 0)
            metrics = np.array(metrics)
            metrics_total += metrics
        except Exception as e:
            print(e)
            print("Error in computing metrics for", audio)
            continue

    metrics_avg = metrics_total / num
    metrics_avg_dict = {
        "pesq": metrics_avg[0].round(3),
        "csig": metrics_avg[1].round(3),
        "cbak": metrics_avg[2].round(3),
        "covl": metrics_avg[3].round(3),
        "ssnr": metrics_avg[4].round(3),
        "stoi": metrics_avg[5].round(3),
    }
    print(noisy_dir, clean_dir, saved_dir)
    print(metrics_avg_dict)
    
    # with open(os.path.join(saved_dir, "_metrics.json"), "w") as f:
    #     json.dump(metrics_avg_dict, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_dir", type=str, required=True, help="Directory containing the original noisy data")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory containing the original clean data")
    parser.add_argument("--enhanced_dir", type=str, required=True, help="Directory containing the enhanced data")
    parser.add_argument("--target_sr", type=int, default=None, help="Target sample rate for resampling")
    
    args = parser.parse_args()

    evaluate_from_audio(args.noisy_dir, args.clean_dir, args.enhanced_dir, args.target_sr)