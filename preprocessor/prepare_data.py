import os
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

def prepare_data(data_dir, corpus_name, corpus_dir=None, n_person=None, use_nsml=False):
    print("prepare data...")
    # NSML
    if use_nsml:
        import nsml
        from nsml import DATASET_PATH
        corpus_dir = Path(DATASET_PATH) / "train"
    else:
        corpus_dir = Path(corpus_dir)

    if corpus_name == "JVS":
        speaker_dirs = os.listdir(corpus_dir)
        speaker_dirs = [
            corpus_dir / d 
            for d in speaker_dirs if os.path.isdir(os.path.join(corpus_dir, d))
        ][
            :n_person
        ]

        utt_list = []
        for speaker_dir in tqdm(speaker_dirs, desc="prepare data..."):
            for data_type in ["parallel100", "nonpara30"]:

                wav_dir = Path(speaker_dir / data_type / "wav24kHz16bit")
                utt_list += [
                    ":".join([speaker_dir.name, data_type, d.split(".")[0]]) 
                    for d in os.listdir(wav_dir) if d.endswith(".wav")
                ]

    elif corpus_name == "JSUT":
        trn_file = corpus_dir / "basic5000" / "transcript_utf8.txt"
        with open(trn_file, "r") as f:
            utt_list = [
                l.strip().split(":")[0] 
                for l in f if l.strip().split(":")[0] != "BASIC5000_3461"]

    utt_list_path = Path(data_dir) / "utt.list"
    with open(utt_list_path, "w") as f:
        f.write("\n".join(utt_list))

@hydra.main(config_path="conf/preprocess", config_name="config")
def main(config: DictConfig):

    prepare_data(config)

if __name__ == "__main__":
    main()