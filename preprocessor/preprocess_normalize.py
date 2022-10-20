import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import joblib

import hydra
from omegaconf import DictConfig

import numpy as np


def normalize(path, scaler, out_dir):
    x = np.load(path).T
    y = scaler.transform(x)
    if np.isnan(y).sum() > 0:
        print("nan is found in {}".format(str(out_dir)))
    np.save(out_dir / path.name, y.T)


def preprocess_normalize(data_dir, utt_list_path, scaler_data_dir,
                         remove_input=False, n_jobs=4):
    print("normalize features...")

    for feat in ["spectrogram", "f0", "spectrum", "aperiodicity"]:

        with open(utt_list_path) as f:
            utt_list = [utt.strip() for utt in f]
        in_dir = data_dir / "org" / feat
        out_dir = data_dir / "norm" /  feat 
        paths = [
            Path(in_dir / "{}-feats.npy".format(utt.replace(":", "-"))) 
            for utt in utt_list]
        scaler = joblib.load(scaler_data_dir / "norm" / feat / "scaler.joblib")

        out_dir.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(n_jobs) as executor:
            futures = [
                executor.submit(normalize, path, scaler, out_dir)
                for path in paths]
            for future in tqdm(futures, desc=feat):
                future.result()

        # 正規化前のデータを削除
        if remove_input:
            for p in in_dir.glob("*-feats.npy"):
                if p.is_file():
                    os.remove(p)

@hydra.main(config_path="conf/preprocess", config_name="config")
def main(config: DictConfig):

    preprocess_normalize(config)

if __name__ == "__main__":
    main()
