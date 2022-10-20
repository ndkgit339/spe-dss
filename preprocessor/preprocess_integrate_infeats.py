from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import numpy as np

def process(f0_path, sp_path, ap_path, vuv_path, inout_dir):
    f0 = np.load(f0_path)
    sp = np.load(sp_path)
    ap = np.load(ap_path)
    vuv = np.load(vuv_path)
    
    in_feature = np.concatenate([f0, sp, ap, vuv], axis=1)

    np.save(inout_dir / f0_path.name, in_feature)

def preprocess_integrate_infeats(config):
    print("integrate input features...")

    data_dir = Path(config.data_dir)
    for phase in ["train", "dev", "eval"]:

        utt_list_name = f"{phase}_frame.list" if config.extract_frame else f"{phase}.list"
        with open(data_dir / utt_list_name) as f:
            utt_list = [utt.strip() for utt in f]

        in_dir = data_dir / "norm" / phase / "in_feats"
        in_dir_f0 = in_dir / "f0"
        in_dir_sp = in_dir / "spectrum"
        in_dir_ap = in_dir / "aperiodicity"
        in_dir_vuv = in_dir / "vuv"

        f0_paths = [Path(in_dir_f0) / "{}-feats.npy".format(utt.replace(":", "-")) for utt in utt_list]
        sp_paths = [Path(in_dir_sp) / "{}-feats.npy".format(utt.replace(":", "-")) for utt in utt_list]
        ap_paths = [Path(in_dir_ap) / "{}-feats.npy".format(utt.replace(":", "-")) for utt in utt_list]
        vuv_paths = [Path(in_dir_vuv) / "{}-feats.npy".format(utt.replace(":", "-")) for utt in utt_list]

        with ProcessPoolExecutor(config.n_jobs) as executor:
            futures = [
                executor.submit(
                    process,
                    f0_path,
                    sp_path,
                    ap_path,
                    vuv_path,
                    in_dir,
                )
                for f0_path, sp_path, ap_path, vuv_path in zip(f0_paths, sp_paths, ap_paths, vuv_paths)
            ]
            for future in tqdm(futures):
                future.result()

        # # 統合前のデータを削除
        # for in_dir_feat in [in_dir_f0, in_dir_sp, in_dir_ap, in_dir_vuv]:
        #     for p in in_dir_feat.glob("*-feats.npy"):
        #         if p.is_file():
        #             os.remove(p)

@hydra.main(config_path="conf/preprocess", config_name="config")
def main(config: DictConfig):

    preprocess_integrate_infeats(config)

if __name__ == "__main__":
    main()
