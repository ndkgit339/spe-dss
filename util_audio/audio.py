import matplotlib.pyplot as plt
import numpy as np
import torch


def interpolate_lin_to_lin(src_spectrogram, dst_spec_size, sample_rate, device="cpu"):
    """
    Parameters
    ----------
    src_spectrogram: torch.Tensor
        shape (n_batch, n_freq, n_frame)
    dst_spec_size: int
    sample_rate: int
    device: torch.device

    Returns
    -------
    dst_spectrogram: torch.Tensor
        shape (n_batch, n_freq, n_frame)
    """

    max_hz = int(sample_rate / 2)

    src_spec_size = src_spectrogram.size(1)
    if src_spec_size == dst_spec_size:
        return src_spectrogram

    f = torch.arange(0, dst_spec_size).to(device)
    dst_x = f / (dst_spec_size - 1)
    dst_hz = dst_x * max_hz
    src_x = torch.clamp(dst_hz / max_hz, max=1)
    src_f_center = src_x * (src_spec_size - 1)
    src_f_left = torch.floor(src_f_center).to(torch.int)
    src_f_right = torch.ceil(src_f_center).to(torch.int)
    src_f_right_ratio = src_f_center - src_f_left
    src_spectrogram_left = torch.index_select(src_spectrogram, 1, src_f_left)
    src_spectrogram_right = torch.index_select(src_spectrogram, 1, src_f_right)
    dst_spectrogram = src_spectrogram_left.transpose(1, 2) + src_f_right_ratio * (
        src_spectrogram_right.transpose(1, 2) - src_spectrogram_left.transpose(1, 2)
    )
    return dst_spectrogram.transpose(1, 2)


def initialize_aperiodicity(n_batch, fft_size, n_frame):
    return torch.tensor([1.0 - 1e-12]).expand(
        n_batch, int(fft_size / 2) + 1, n_frame
    )


def get_number_of_aperiodicities(fs):
    return int(min(15000.0, fs / 2.0 - 3000.0) / 3000.0)


def get_decoding_filter(fs, fft_size, n_ap):

    freq_axis = [fs / fft_size * i for i in range(int(fft_size / 2) + 1)]
    coarse_freq_axis = [3000.0 * i for i in range(n_ap + 1)] + [fs / 2.0]

    decoding_filter = torch.zeros(int(fft_size / 2) + 1, n_ap + 2)
    i_c = 0
    i = 0
    while i < int(fft_size / 2) + 1 and i_c < n_ap + 1:
        if freq_axis[i] >= coarse_freq_axis[i_c + 1]:
            i_c += 1
            continue
        diff = coarse_freq_axis[i_c + 1] - coarse_freq_axis[i_c]
        decoding_filter[i][i_c] = (coarse_freq_axis[i_c + 1] - freq_axis[i]) / diff
        decoding_filter[i][i_c + 1] = (freq_axis[i] - coarse_freq_axis[i_c]) / diff
        i += 1

    return decoding_filter


def torch_decode_aperiodicity(coded_aperiodicity, fs, fft_size, device=None):

    if device is None:
        device = coded_aperiodicity.device

    n_batch = coded_aperiodicity.shape[0]
    n_frame = coded_aperiodicity.shape[2]

    init_ap = initialize_aperiodicity(n_batch, fft_size, n_frame).to(device)
    n_ap = get_number_of_aperiodicities(fs)

    coarse_ap_first = torch.tensor([-60.0]).expand(n_batch, 1, n_frame).to(device)
    coarse_ap_last = torch.tensor([-1e-12]).expand(n_batch, 1, n_frame).to(device)
    coarse_ap = torch.cat([
        coarse_ap_first,
        coded_aperiodicity,
        coarse_ap_last,
    ], dim=1)

    decoding_filter = get_decoding_filter(
        fs, fft_size, n_ap
    ).unsqueeze(0).expand(n_batch, -1, -1).to(device)
    decoded_aperiodicity = torch.bmm(decoding_filter, coarse_ap)

    decoded_aperiodicity = torch.where(
        coarse_ap[:, 1:-1, :].mean(dim=1, keepdim=True).expand(-1, int(fft_size / 2) + 1, -1) > -0.5,
        init_ap,
        torch.pow(10.0, decoded_aperiodicity / 20.0),
    )

    return decoded_aperiodicity


def log_spectral_distance(lin_spectrogram_A, lin_spectrogram_B, eps=1e-10, mean=True):

    power_spectrogram_A = torch.square(lin_spectrogram_A)
    power_spectrogram_B = torch.square(lin_spectrogram_B)

    lsd = 20 * torch.log10(power_spectrogram_A.clamp(min=eps) / power_spectrogram_B.clamp(min=eps))

    lsd = lsd * lsd
    lsd = torch.sum(lsd, dim=1)
    lsd = torch.sqrt(lsd)
    if mean:
        return torch.mean(lsd)
    else:
        return lsd.view(-1)


def show_f0(f0s: list, labels: list, frame_shift_ms, width=8):

    timeaxis = [frame_shift_ms * i / 1000 for i in range(len(f0s[0]))]

    if len(f0s) == 1:
        fig, ax = plt.subplots(figsize=(width, 6))
        f0 = f0s[0].detach().cpu().numpy()
        ax.plot(timeaxis, f0, linewidth=2)
        ax.set(title=labels[0], xlabel="Time", ylabel="Hz")
    else:
        fig, ax = plt.subplots(figsize=(width, 6))
        cmap = plt.get_cmap("tab10")
        for i in range(len(f0s)):
            f0 = f0s[i].detach().cpu().numpy()
            ax.plot(timeaxis, f0, linewidth=2, label=labels[i], color=cmap(i % 10))
        ax.set(xlabel="Time", ylabel="Hz")
        plt.legend()
    return fig


def show_spec(
    spectrogram_list, labels, frame_shift_ms=5.0, min_hz=0.0, max_hz=2000.0,
    vmin=None, vmax=None, log=True, width=8, eps=1e-10):

    specs = []
    for spec in spectrogram_list:
        if log:
            spec_ = np.log(spec.detach().cpu().numpy().clip(min=eps))
        else:
            spec_ = spec.detach().cpu().numpy()
        specs.append(spec_)

    rows = len(specs)
    length_sec = specs[0].shape[1] * frame_shift_ms * 0.001
    if vmin is None:
        vmin = np.min(specs[0])
    if vmax is None:
        vmax = np.max(specs[0])

    # Spectrogram
    fig = plt.figure(figsize=(width, 6*rows))
    axes = []
    for i in range(rows):
        spec = specs[i]
        axes.append(fig.add_subplot(rows, 1, i+1))
        axes[-1].set_title(labels[i])
        plt.imshow(
            np.flipud(spec), cmap='jet', aspect="auto",
            vmin=vmin, vmax=vmax,
            extent=[0, length_sec, min_hz, max_hz])

    plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9)
    cax = plt.axes([0.9, 0.1, 0.05, 0.8])
    plt.colorbar(cax=cax)

    return fig
