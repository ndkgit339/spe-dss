import joblib
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from sklearn.preprocessing import StandardScaler

def fit_scaler(data_dir, utt_list_path):
    print("fit scaler...")

    for feat in ["spectrogram", "f0", "spectrum", "aperiodicity"]:

        in_dir = data_dir / "org" / feat
        out_dir = data_dir / "norm" /  feat
        out_dir.mkdir(parents=True, exist_ok=True)

        scaler = StandardScaler()

        with open(utt_list_path, "r") as f:
            utt_list = [utt.strip() for utt in f]

        for utt in tqdm(utt_list, desc=feat):
            c = np.load(in_dir / "{}-feats.npy".format(utt.replace(":", "-"))).T
            scaler.partial_fit(c)

        np.save(out_dir / "scaler_mean_.npy", scaler.mean_.astype(np.float32))
        np.save(out_dir / "scaler_var_.npy", scaler.var_.astype(np.float32))

        joblib.dump(scaler, out_dir / "scaler.joblib")

@hydra.main(config_path="conf/preprocess", config_name="config")
def main(config: DictConfig):

    fit_scaler(config)

if __name__ == "__main__":
    main()
