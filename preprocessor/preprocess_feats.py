import sys, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torchaudio.functional import highpass_biquad

# Audio processing
import librosa
from nnmnkwii.preprocessing.f0 import interp1d
import pyworld as pw

# My library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dss import world_analysis
from util_audio import interpolate_lin_to_lin

def log_f0(f0):

    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    return lf0

def process(utt, wav_file, spec_dir, f0_dir, sp_dir, ap_dir, vuv_dir, wav_dir, 
            sample_rate, n_fft, hop_length, frame_shift_ms, stft_window,
            stft_center, n_coded_sp, n_coded_ap, coding_ap, n_mels_spec,
            comp, melspec=False, highpass=False, eps=1e-10):

    # 入力の計算：F0,包絡,非周期性
    waveform, _ = librosa.load(wav_file, sr=sample_rate)
    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).clone().view([1, -1])

    # ハイパスフィルタ (> 71Hz)
    if highpass:
        waveform_tensor = highpass_biquad(waveform_tensor, 
                                          sample_rate=sample_rate, 
                                          cutoff_freq=70.0)
        waveform = waveform_tensor.squeeze().numpy()

    f0, vuv, sp, ap = world_analysis(waveform_tensor, sample_rate, 
                                         n_fft, frame_shift_ms, 
                                         harvest=True)
    
    # f0 の対数化、線形補完
    lf0 = log_f0(f0.T.numpy().astype(np.float32))
    f0_hz = interp1d(lf0, kind="linear").T
    
    # スペクトル包絡のメル対数化
    if comp:
        spectral_env = interpolate_lin_to_lin(
            torch.log(torch.clamp(sp, min=eps)), 
            dst_spec_size=n_coded_sp, sample_rate=sample_rate
            ).squeeze().numpy()
    else:
        spectral_env = np.log(sp.clip(min=eps)).squeeze()

    # 非周期性指標の低次元化
    if comp:
        if coding_ap == "world":
            aperiodicity = pw.code_aperiodicity(
                    np.ascontiguousarray(ap.squeeze().T.numpy(), dtype=np.float64), 
                    fs=sample_rate,
            ).astype(np.float32).T
        elif coding_ap == "interp":
            aperiodicity = interpolate_lin_to_lin(
                torch.log(torch.clamp(ap, min=eps)), 
                dst_spec_size=n_coded_ap, sample_rate=sample_rate
                ).squeeze().numpy()
    else:
        aperiodicity = np.log(ap.clip(min=eps)).squeeze()

    # VUV
    vuv = vuv.numpy()

    # 出力の計算：振幅スペクトログラム
    if melspec:
        spec = librosa.feature.melspectrogram(
                waveform, n_mels=n_mels_spec, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                win_length=n_fft, window=stft_window, center=stft_center)
    else:
        x = librosa.stft(waveform, n_fft=n_fft, win_length=n_fft, 
                         hop_length=hop_length, window=stft_window, 
                         center=stft_center)
        spec = np.abs(x).astype(np.float32)
    spec = np.log(np.clip(spec, a_min=eps, a_max=None))
    n_frame = spec.shape[1]
    
    # ファイルに保存
    for dir, feat in zip(
        [spec_dir, f0_dir, sp_dir, ap_dir, vuv_dir, wav_dir],
        [spec, f0_hz, spectral_env, aperiodicity, vuv, waveform],
    ):
        if np.isnan(feat).sum() > 0:
            print("nan is found in {}".format(
                str(dir / "{}-feats.npy".format(utt.replace(":", "-")))))
        np.save(
            dir / "{}-feats.npy".format(utt.replace(":", "-")), feat)

    return utt, n_frame

def preprocess_feats(data_dir, utt_list_path, n_frame_list_path, corpus_name,
                     corpus_dir, sample_rate, n_fft, hop_length,
                     frame_shift_ms, stft_window, stft_center, n_coded_sp,
                     n_coded_ap, coding_ap, n_mels_spec, n_frames, comp,
                     melspec, highpass, frames=True, n_jobs=4):

    utt_list = []
    wav_file_paths = []
    if frames:
        wav_file_paths = []
        with open(utt_list_path, "r") as f:
            for utt in f:
                if len(utt.strip()) > 0:
                    utt = utt.strip()
                    utt_list.append(utt)
                    wav_file_paths.append(data_dir / "wav_{}frames".format(n_frames) / "{}.wav".format(utt.replace(":", "-")))
    else:
        with open(utt_list_path, "r") as f:
            for utt in f:
                if len(utt.strip()) > 0:
                    utt = utt.strip()
                    utt_list.append(utt)
                    if corpus_name == "JVS":
                        speaker, data_type, utt_id = tuple(utt.split(":"))
                        wav_file_paths.append(
                            corpus_dir / speaker / data_type / "wav24kHz16bit" / f"{utt_id}.wav"
                        )
                    elif corpus_name == "JSUT":
                        wav_file_paths.append(corpus_dir / "basic5000" / "wav" / f"{utt}.wav")

    spec_dir = data_dir / "org" / "spectrogram"
    f0_dir = data_dir / "org" / "f0"
    sp_dir = data_dir / "org" / "spectrum"
    ap_dir = data_dir / "org" / "aperiodicity"
    vuv_dir = data_dir / "org" / "vuv"
    wav_dir = data_dir / "org" / "wav"

    for d in [spec_dir, f0_dir, sp_dir, ap_dir, vuv_dir, wav_dir]:
        d.mkdir(parents=True, exist_ok=True)

    output = []
    with ProcessPoolExecutor(n_jobs) as executor:
        futures = [
            executor.submit(
                process,
                utt,
                wav_file_path,
                spec_dir,
                f0_dir,
                sp_dir,
                ap_dir,
                vuv_dir,
                wav_dir,
                sample_rate,
                n_fft,
                hop_length,
                frame_shift_ms,
                stft_window,
                stft_center,
                n_coded_sp,
                n_coded_ap,
                coding_ap,
                n_mels_spec,
                comp,
                melspec,
                highpass,
            )
            for utt, wav_file_path in zip(utt_list, wav_file_paths)
        ]
        bar = tqdm(total=len(futures), desc="preprocess features...")
        for future in futures:
            utt, n_frame = future.result()
            output.append("{}:{}".format(utt.replace(":", "-"), n_frame))
            bar.update(1)
    
    with open(n_frame_list_path, "w") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    pass
