from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import librosa
import soundfile as sf

def preprocess_wav(config, audio_params, use_nsml=False):
    # NSML
    if use_nsml:
        import nsml
        from nsml import DATASET_PATH
        corpus_dir = Path(DATASET_PATH) / "train"
    else:
        corpus_dir = Path(config.corpus_dir)

    # Corpus name
    corpus_name = config.corpus_name

    data_dir = Path(config.data_dir)

    sample_rate = audio_params["sample_rate"]
    hop_length = audio_params["hop_length"]
    n_frames = config.n_frames
    n_samples = (n_frames - 1) * hop_length

    out_dir = data_dir / f"wav_{n_frames}frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read utt list
    with open(data_dir / "train.list", "r") as f:
        utt_list = [l.strip() for l in f if len(l.strip()) > 0]

    # Extract frames
    out_utt_list = []
    bar = tqdm(total=len(utt_list), desc="preprocess wav...")
    for utt in utt_list:
        if corpus_name == "JVS":
            wav_path = corpus_dir / utt.split(":")[0] / utt.split(":")[1] / "wav24kHz16bit" / "{}.wav".format(utt.split(":")[2])
        elif corpus_name == "JSUT":
            wav_path = corpus_dir / "basic5000" / "wav" / "{}.wav".format((utt))

        waveform, _ = librosa.load(wav_path, sr=sample_rate)

        for i in range(len(waveform) // n_samples):
            out_waveform = waveform[i * n_samples:(i + 1) * n_samples]
            sf.write(out_dir / (utt.replace(":", "-") + f"-{i}.wav"), out_waveform, sample_rate)
            out_utt_list.append(f"{utt}:{i}")

        bar.update(1)

    # Save frame-extracted utt list
    with open(data_dir / f"train_frame.list", "w") as f:
        f.write("\n".join(out_utt_list))

@hydra.main(config_path="conf/preprocess", config_name="config")
def main(config: DictConfig):
    preprocess_wav(config)

if __name__ == "__main__":

    main()
