import argparse
import glob
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from dss import DifferentiableSpeechSynthesizer, world_analysis, world_synthesis


def main():

    # ArgumentParser
    parser = argparse.ArgumentParser(description="")

    # Arguments
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device name. Example: "cpu", "cuda", "cuda:0", "cuda:1" Default = "cuda"',
    )
    parser.add_argument(
        "--input_data_dir",
        required=True,
        help="Directory containing wav data to be input",
    )
    parser.add_argument(
        "--output_data_dir", required=True, help="Directory to output synthesis results"
    )
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    parser.add_argument(
        "--analysis_frame_shift_ms",
        type=float,
        default=5.0,
        help="Frame shift time for analysis [ms]",
    )
    parser.add_argument(
        "--synthesis_frame_shift_ms",
        type=float,
        default=1.0,
        help="Frame shift time for synthesis [ms]",
    )
    args = parser.parse_args()

    # Device
    device_name = args.device
    if torch.cuda.is_available():
        device = torch.device(str(device_name))
    else:
        device = torch.device("cpu")

    # Input, Output
    input_dir = args.input_data_dir
    output_dir = args.output_data_dir
    os.makedirs(output_dir, exist_ok=True)

    # Seed
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # FFT Size, Frame Shift
    n_fft = args.n_fft
    analysis_frame_shift_ms = args.analysis_frame_shift_ms
    synthesis_frame_shift_ms = args.synthesis_frame_shift_ms

    # Wav
    wav_files_list = sorted(
        glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    )
    for idx, filename in enumerate(wav_files_list):

        # Print
        print("[" + str(idx + 1) + "/" + str(len(wav_files_list)) + "]", filename)

        # Load Wav & Resample
        waveform, sample_rate = librosa.load(filename, sr=None)
        waveform_tensor = (
            torch.from_numpy(waveform.astype(np.float32)).clone().view([1, -1])
        )

        # Hop Length
        hop_length = int(sample_rate * analysis_frame_shift_ms / 1000.0)
        synth_hop_length = int(sample_rate * synthesis_frame_shift_ms / 1000.0)

        # Get DSS
        synthesizer = DifferentiableSpeechSynthesizer(
            device=device,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            synth_hop_length=synth_hop_length,
        )

        # World Parameter
        f0, vuv, sp, ap = world_analysis(
            waveform=waveform_tensor,
            fs=sample_rate,
            n_fft=n_fft,
            frame_period=analysis_frame_shift_ms,
            harvest=True,
        )

        # Synth by World
        world_wav = world_synthesis(
            f0=f0, sp=sp, ap=ap, fs=sample_rate, frame_period=analysis_frame_shift_ms
        )

        # NOTE: When using the synthesis process by DSS, determine the lower limit
        # of F0 or convert to continuous F0. If F0 = 0[Hz], it will not work well.
        f0 = torch.clamp(input=f0, min=71.0)

        # Synth by DSS
        dss_wav = synthesizer(f0_hz=f0, spectral_env=sp, aperiodicity=ap)

        # Output Audio File
        dss_wav = dss_wav[0].cpu().squeeze().detach().numpy().copy()
        world_wav = world_wav.cpu().numpy()
        base_name = os.path.splitext(os.path.basename(filename))[0]
        os.makedirs(output_dir, exist_ok=True)
        sf.write(
            file=os.path.join(output_dir, base_name + "_world.wav"),
            data=world_wav,
            samplerate=sample_rate,
        )
        sf.write(
            file=os.path.join(output_dir, base_name + "_dss.wav"),
            data=dss_wav,
            samplerate=sample_rate,
        )


if __name__ == "__main__":
    main()
