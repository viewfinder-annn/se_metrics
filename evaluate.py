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
        est_audio, sr = sf.read(saved_path)
        clean_audio, sr = sf.read(clean_path)
        # Resample to target_sr
        if target_sr is not None:
            est_audio = torchaudio.transforms.Resample(sr, target_sr)(torch.tensor(est_audio).unsqueeze(0)).squeeze().numpy()
            clean_audio = torchaudio.transforms.Resample(sr, target_sr)(torch.tensor(clean_audio).unsqueeze(0)).squeeze().numpy()
        else:
            target_sr = sr
        metrics = compute_metrics(clean_audio, est_audio, target_sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    metrics_avg_dict = {
        "pesq": metrics_avg[0],
        "csig": metrics_avg[1],
        "cbak": metrics_avg[2],
        "covl": metrics_avg[3],
        "ssnr": metrics_avg[4],
        "stoi": metrics_avg[5],
    }
    
    print(metrics_avg_dict)
    
    with open(os.path.join(saved_dir, "_metrics.json"), "w") as f:
        json.dump(metrics_avg_dict, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_dir", type=str, required=True, help="Directory containing the original noisy data")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory containing the original clean data")
    parser.add_argument("--enhanced_dir", type=str, required=True, help="Directory containing the enhanced data")
    parser.add_argument("--target_sr", type=int, default=None, help="Target sample rate for resampling")
    
    args = parser.parse_args()

    evaluate_from_audio(args.noisy_dir, args.clean_dir, args.enhanced_dir, args.target_sr)