import os
import random
import sys

import numpy as np
import torch
from parallel_wavegan.losses import stft

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dss import world_analysis, world_synthesis
from util_audio import (interpolate_lin_to_lin, log_spectral_distance,
                        torch_decode_aperiodicity)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
def get_scaler_vals(data_dir):

    scaler_means = {}
    scaler_vars = {}
    scaler_means["f0"] = torch.from_numpy(np.load(
        data_dir / "norm" / "f0" / "scaler_mean_.npy"
        ).astype(np.float32))
    scaler_vars["f0"] = torch.from_numpy(np.load(
        data_dir / "norm" / "f0" / "scaler_var_.npy"
        ).astype(np.float32))
    for feat in ["spectrogram", "spectrum", "aperiodicity"]:
        scaler_means[feat] = torch.from_numpy(np.load(
            data_dir / "norm" / feat / "scaler_mean_.npy"
            ).astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=2)
        scaler_vars[feat] = torch.from_numpy(np.load(
            data_dir / "norm" / feat / "scaler_var_.npy"
            ).astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=2)

    return scaler_means, scaler_vars


def decode_feats(log_norm_f0_hz, log_norm_mel_sp, log_norm_bap,
                 scaler_means, scaler_vars, sample_rate, n_fft, 
                 comp, coding_ap, device, eps=1e-10):

    # 非標準化        
    log_f0_hz = log_norm_f0_hz.squeeze(1).to(device) * torch.sqrt(
        torch.clamp(scaler_vars["f0"].to(device), min=eps)) + scaler_means["f0"].to(device)
    log_mel_sp = log_norm_mel_sp.to(device) * torch.sqrt(
        torch.clamp(scaler_vars["spectrum"].to(device), min=eps)) + scaler_means["spectrum"].to(device)
    log_bap = log_norm_bap.to(device) * torch.sqrt(
        torch.clamp(scaler_vars["aperiodicity"].to(device), min=eps)) + scaler_means["aperiodicity"].to(device)

    # f0（非対数化、71Hz以上）
    f0_hz = torch.clamp(torch.exp(torch.clamp(log_f0_hz, max=10.0)),
                        min=71.0, max=2000.0)

    # スペクトル包絡（線形補完
    if comp:
        spectral_env = torch.clamp(
            torch.exp(torch.clamp(
                interpolate_lin_to_lin(
                    log_mel_sp, dst_spec_size=int(n_fft / 2) + 1, 
                    sample_rate=sample_rate, device=device),
                max=10.0)),
            min=0.0, max=10.0)
    else:
        spectral_env = torch.clamp(
            torch.exp(torch.clamp(log_mel_sp, max=10.0)),
            min=0.0, max=10.0)

    # 非周期性指標（バンドから復元）
    if comp:
        if coding_ap == "world":
            aperiodicity = torch.clamp(
                torch_decode_aperiodicity(
                    log_bap, fs=sample_rate, fft_size=n_fft, device=device),
                min=0.0, max=1.0)
        elif coding_ap == "interp":
            aperiodicity = torch.clamp(
                torch.exp(torch.clamp(
                    interpolate_lin_to_lin(
                        log_bap, dst_spec_size=int(n_fft / 2) + 1, 
                        sample_rate=sample_rate, device=device),
                    max=0.0)),
                min=0.0, max=1.0)
    else:
        aperiodicity = torch.clamp(torch.exp(torch.clamp(log_bap, max=0.0)), 
                                   min=0.0, max=1.0)

    return f0_hz, spectral_env, aperiodicity


def world_analyze_synth(device, waveform, sample_rate, n_fft,
                        hop_length, harvest=True):

    frame_shift_ms = 1000.0 * hop_length / sample_rate

    f0, _, sp, ap = world_analysis(waveform.to(device), sample_rate,
                                   n_fft, frame_shift_ms, harvest=harvest)
    waveform_syn = world_synthesis(f0, sp, ap, sample_rate,
                                    frame_period=frame_shift_ms,
                                    ).unsqueeze(0).to(device)
    return waveform_syn[ : , : waveform.shape[0]]


def synth_from_coded_feats(speech_synthesizer, f0, spectrum, aperiodicity,
                           scaler_means, scaler_vars, n_fft, sample_rate,
                           device, comp=False, coding_ap="world"):

    f0_coded, spectrum_coded, aperiodicity_coded = decode_feats(
        f0, spectrum, aperiodicity, scaler_means, scaler_vars,
        sample_rate, n_fft, comp, coding_ap, device)
    
    # Synthesize
    waveform_syn = speech_synthesizer(
        torch.clamp(f0_coded.squeeze(1), min=71.0, max=2000.0),
        torch.clamp(spectrum_coded, min=0.0, max=10.0), 
        torch.clamp(aperiodicity_coded, min=0.0, max=1.0)).to(device)

    return f0_coded, spectrum_coded, aperiodicity_coded, waveform_syn


def calc_lsd(device, speech_synthesizer, scaler_means, scaler_vars,
             sample_rate, n_fft, hop_length, frame_shift_ms, lsd_fft_size,
             lsd_shift_size, lsd_win_length, comp, coding_ap, wav_target,
             f0_target=None, spectrum_target=None, aperiodicity_target=None,
             wav_hat=None, f0_hat=None, spectrum_hat=None, aperiodicity_hat=None,
             lsd_cut_edge=1024, mean=True, use_generator=True):

    with torch.no_grad():
        wav_target = wav_target.to(device)

        if use_generator:
            wav_hat = wav_hat.to(device)
            gen_world_wave = world_synthesis(
                f0_hat, spectrum_hat, aperiodicity_hat, fs=sample_rate,
                frame_period=frame_shift_ms).unsqueeze(0).to(device)
        else:
            _, _, _, pytorch_waveform_syn_coded = synth_from_coded_feats(
                speech_synthesizer, f0_target, spectrum_target,
                aperiodicity_target, scaler_means, scaler_vars,
                n_fft, sample_rate, device, comp=comp, coding_ap=coding_ap)

            world_world_wave_list = []
            world_pytorch_wave_list = []
            for i in range(wav_target.shape[0]):
                world_world_wave_list.append(world_analyze_synth(
                    device, wav_target[i], sample_rate, n_fft, hop_length))
                world_pytorch_wave_list.append(world_analyze_synth(
                    device, wav_target[i], sample_rate, n_fft, hop_length))
            world_world_wave = torch.cat(world_world_wave_list, dim=0)
            world_pytorch_wave = torch.cat(world_pytorch_wave_list, dim=0)

        if use_generator:
            gen_world_magspec = stft(
                gen_world_wave[ : , lsd_cut_edge : - lsd_cut_edge], 
                lsd_fft_size, lsd_shift_size, lsd_win_length,
                torch.hann_window(lsd_win_length).to(device)).transpose(1, 2)
            gen_pytorch_magspec = stft(
                wav_hat[ : , lsd_cut_edge : - lsd_cut_edge], 
                lsd_fft_size, lsd_shift_size, lsd_win_length,
                torch.hann_window(lsd_win_length).to(device)).transpose(1, 2)
        else:
            world_coded_pytorch_magspec = stft(
                pytorch_waveform_syn_coded[ : , lsd_cut_edge : - lsd_cut_edge],
                lsd_fft_size, lsd_shift_size, lsd_win_length,
                torch.hann_window(lsd_win_length).to(device)).transpose(1, 2)
            world_world_magspec = stft(
                world_world_wave[ : , lsd_cut_edge : - lsd_cut_edge], 
                lsd_fft_size, lsd_shift_size, lsd_win_length,
                torch.hann_window(lsd_win_length).to(device)).transpose(1, 2)
            world_pytorch_magspec = stft(
                world_pytorch_wave[ : , lsd_cut_edge : - lsd_cut_edge], 
                lsd_fft_size, lsd_shift_size, lsd_win_length,
                torch.hann_window(lsd_win_length).to(device)).transpose(1, 2)
        target_magspec = stft(
            wav_target[ : , lsd_cut_edge : - lsd_cut_edge], 
            lsd_fft_size, lsd_shift_size, lsd_win_length,
            torch.hann_window(lsd_win_length).to(device)).transpose(1, 2)

        if use_generator:
            gen_world_lsd = log_spectral_distance(
                gen_world_magspec[ : , : , : target_magspec.shape[2]], 
                target_magspec, mean=mean)
            gen_pytorch_lsd = log_spectral_distance(
                gen_pytorch_magspec[ : , : , : target_magspec.shape[2]], 
                target_magspec, mean=mean)
        else:
            world_coded_pytorch_lsd = log_spectral_distance(
                world_coded_pytorch_magspec[ : , : , : target_magspec.shape[2]], 
                target_magspec, mean=mean)
            world_world_lsd = log_spectral_distance(
                world_world_magspec[ : , : , : target_magspec.shape[2]], 
                target_magspec, mean=mean)
            world_pytorch_lsd = log_spectral_distance(
                world_pytorch_magspec[ : , : , : target_magspec.shape[2]], 
                target_magspec, mean=mean)

    if use_generator:
        if mean:    
            return gen_world_lsd, gen_pytorch_lsd
        else:
            return gen_world_lsd.view(-1), gen_pytorch_lsd.view(-1)
    else:
        if mean:    
            return world_coded_pytorch_lsd, world_world_lsd, world_pytorch_lsd
        else:
            return world_coded_pytorch_lsd.view(-1), \
                   world_world_lsd.view(-1), world_pytorch_lsd.view(-1)
