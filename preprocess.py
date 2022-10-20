import time
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig

# My library
from preprocessor import prepare_data, split_data, preprocess_feats, \
                         fit_scaler, preprocess_normalize
from util_train import set_seed


@hydra.main(config_path="conf", config_name="config")
def preprocess(config: DictConfig):

    t_start = time.time()

    corpus_dir = Path(config.preprocess.corpus_dir)

    data_dir = Path(config.preprocess.data_dir)

    utt_list_path = data_dir / "utt.list"
    utt_list_path_train = data_dir / "train.list"
    utt_list_path_dev = data_dir / "dev.list"
    utt_list_path_synth = data_dir / "synth.list"
    utt_list_path_eval = data_dir / "eval.list"
    utt_list_path_traindeveval = data_dir / "traindevsyntheval.list"

    corpus_name = config.preprocess.corpus_name
    n_person = config.preprocess.n_person if corpus_name == "JVS" else None
    split_rates = config.preprocess.split_rates
    n_frames = config.preprocess.n_frames
    comp = config.preprocess.comp
    melspec = config.preprocess.melspec
    highpass = config.preprocess.highpass
    n_jobs = config.preprocess.n_jobs

    sample_rate = config.preprocess.audio.sample_rate
    n_fft = config.preprocess.audio.n_fft
    hop_length = config.preprocess.audio.hop_length
    frame_shift_ms = 1000.0 * hop_length / sample_rate
    stft_window = config.preprocess.audio.stft_window
    stft_center = config.preprocess.audio.stft_center
    n_coded_sp = config.preprocess.audio.n_coded_sp
    n_coded_ap = config.preprocess.audio.n_coded_ap
    coding_ap = config.preprocess.audio.coding_ap
    n_mels_spec = config.preprocess.audio.n_mels_spec

    # Save config
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)


    set_seed(config.seed)

    prepare_data(data_dir, corpus_name, corpus_dir, n_person)
    split_data(utt_list_path, utt_list_path_train, utt_list_path_dev,
               utt_list_path_synth, utt_list_path_eval,
               utt_list_path_traindeveval, corpus_name, split_rates)

    n_frame_list_path = data_dir / "traindeveval_n_frame.list"
    preprocess_feats(data_dir, utt_list_path_traindeveval, n_frame_list_path, 
                     corpus_name, corpus_dir, sample_rate, n_fft, hop_length,
                     frame_shift_ms, stft_window, stft_center, n_coded_sp,
                     n_coded_ap, coding_ap, n_mels_spec, n_frames, comp, 
                     melspec, highpass, frames=False, n_jobs=n_jobs)

    fit_scaler(data_dir, utt_list_path_train)

    preprocess_normalize(data_dir, utt_list_path_train, data_dir, n_jobs=n_jobs)
    preprocess_normalize(data_dir, utt_list_path_dev, data_dir, n_jobs=n_jobs)
    preprocess_normalize(data_dir, utt_list_path_synth, data_dir, n_jobs=n_jobs)
    preprocess_normalize(data_dir, utt_list_path_eval, data_dir, n_jobs=n_jobs)

    t_end = time.time() - t_start
    with open(data_dir / "time.log", "w") as f:
        f.write(f"preprocess time: {str(t_end)} s")


if __name__ == "__main__":
    preprocess()