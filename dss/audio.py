import numpy as np
import pyworld as pw
import torch


def world_analysis(
    waveform, fs, n_fft, frame_period=5.0, f0_ceil=1000.0, threshold=0.85, harvest=True
):
    waveform_nd = waveform.squeeze().cpu().numpy().astype(np.float64)
    if harvest:
        f0, t = pw.harvest(waveform_nd, fs, frame_period=frame_period, f0_ceil=f0_ceil)
    else:
        f0, t = pw.dio(waveform_nd, fs, frame_period=frame_period, f0_ceil=f0_ceil)
        f0 = pw.stonemask(waveform_nd, f0, t, fs)
    sp = pw.cheaptrick(waveform_nd, f0, t, fs, fft_size=n_fft)
    ap = pw.d4c(waveform_nd, f0, t, fs, fft_size=n_fft, threshold=threshold)

    f0 = torch.from_numpy(f0.astype(np.float32)).clone()
    sp = torch.from_numpy(sp.astype(np.float32)).clone()
    ap = torch.from_numpy(ap.astype(np.float32)).clone()
    sp = sp.transpose(0, 1)
    ap = ap.transpose(0, 1)
    f0 = torch.unsqueeze(f0, 0)
    sp = torch.unsqueeze(sp, 0)
    ap = torch.unsqueeze(ap, 0)

    if harvest:
        vuv = torch.where(ap[:, 0, :] < 0.5, torch.ones(1), torch.zeros(1))
    else:
        vuv = torch.where(f0 > 0, torch.ones(1), torch.zeros(1))
    return f0, vuv, sp, ap


def world_synthesis(f0, sp, ap, fs, frame_period=5.0):
    f0 = f0[0].cpu().detach().numpy().astype(np.float64).copy(order="C")
    sp = sp[0].transpose(0, 1).cpu().detach().numpy().astype(np.float64).copy(order="C")
    ap = ap[0].transpose(0, 1).cpu().detach().numpy().astype(np.float64).copy(order="C")

    waveform = pw.synthesize(f0, sp, ap, fs, frame_period=frame_period)
    waveform = torch.from_numpy(waveform.astype(np.float32)).clone()
    return waveform
