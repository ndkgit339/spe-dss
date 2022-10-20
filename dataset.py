import numpy as np

import torch
from torch.utils.data import Dataset

def pad_1d(x, max_len, mode="constant", constant_values=0):
    x = np.pad(
        x,
        (0, max_len - x.shape[0]),
        mode=mode, constant_values=constant_values)
    return x

def pad_2d(x, max_len, mode="constant", constant_values=0):
    x = np.pad(
        x,
        [(0, 0), (0, max_len - x.shape[1])],
        mode=mode, constant_values=constant_values)
    return x

class MyDataset(Dataset):
    def __init__(self, phase, spectrogram_paths, f0_paths, spectrum_paths,
                 aperiodicity_paths, wav_paths, start_frames=None,
                 n_frames=400, hop_length=120):

        self.phase = phase

        self.spectrogram_paths = spectrogram_paths
        self.f0_paths = f0_paths
        self.spectrum_paths = spectrum_paths
        self.aperiodicity_paths = aperiodicity_paths
        self.wav_paths = wav_paths

        self.start_frames = start_frames

        self.n_frames = n_frames
        self.hop_length = hop_length

    def __getitem__(self, index):

        # get index
        if self.phase == "train" or self.phase == "dev":
            index, start_frame = self.start_frames[index]

        # get basename
        feats = {}
        feats["basename"] = self.f0_paths[index].stem.replace("-feats", "")

        # get features
        if self.phase == "train" or self.phase == "dev":
            feats["spectrogram"] = np.load(self.spectrogram_paths[index])[
                :, start_frame : start_frame + self.n_frames
                ].astype(np.float32)
            feats["f0"] = np.load(self.f0_paths[index])[
                :, start_frame : start_frame + self.n_frames
                ].astype(np.float32)
            feats["spectrum"] = np.load(self.spectrum_paths[index])[
                :, start_frame : start_frame + self.n_frames
                ].astype(np.float32)
            feats["aperiodicity"] = np.load(self.aperiodicity_paths[index])[
                :, start_frame : start_frame + self.n_frames
                ].astype(np.float32)
            feats["wav"] = np.load(self.wav_paths[index])[
                start_frame * self.hop_length : (start_frame + self.n_frames) * self.hop_length
                ].astype(np.float32)
            if np.isnan(feats["spectrogram"]).sum() > 0:
                print("{} nan is found in {}".format("spectrogram", feats["basename"]))
            if np.isnan(feats["f0"]).sum() > 0:
                print("{} nan is found in {}".format("f0", feats["basename"]))
            if np.isnan(feats["spectrum"]).sum() > 0:
                print("{} nan is found in {}".format("spectrum", feats["basename"]))
            if np.isnan(feats["aperiodicity"]).sum() > 0:
                print("{} nan is found in {}".format("aperiodicity", feats["basename"]))
            if np.isnan(feats["wav"]).sum() > 0:
                print("{} nan is found in {}".format("wav", feats["basename"]))                                                
        elif self.phase == "synth":
            feats["spectrogram"] = np.load(
                self.spectrogram_paths[index]).astype(np.float32)
            feats["f0"] = np.load(
                self.f0_paths[index]).astype(np.float32)
            feats["spectrum"] = np.load(
                self.spectrum_paths[index]).astype(np.float32)
            feats["aperiodicity"] = np.load(
                self.aperiodicity_paths[index]).astype(np.float32)
            feats["wav"] = np.load(
                self.wav_paths[index]).astype(np.float32) 
        # print(feats["f0"].shape)
        return feats

    def __len__(self):
        if self.phase == "train" or self.phase == "dev":
            return len(self.start_frames)
        elif self.phase == "synth":
            return len(self.f0_paths)

    def collate_fn(self, datas):
        basenames = [d["basename"] for d in datas]

        max_len_feats = max([d["spectrogram"].shape[1] for d in datas])
        spectrogram = torch.stack([
            torch.from_numpy(pad_2d(
                d["spectrogram"], max_len_feats, constant_values=0))
            for d in datas
        ], dim=0)
        f0 = torch.stack([
            torch.from_numpy(pad_2d(
                d["f0"], max_len_feats, constant_values=0))
            for d in datas
        ], dim=0)
        spectrum = torch.stack([
            torch.from_numpy(pad_2d(
                d["spectrum"], max_len_feats, constant_values=0))
            for d in datas
        ], dim=0)
        aperiodicity = torch.stack([
            torch.from_numpy(pad_2d(
                d["aperiodicity"], max_len_feats, constant_values=0))
            for d in datas
        ], dim=0)

        max_len_wav = max([len(d["wav"]) for d in datas])
        wav = torch.stack([
            torch.from_numpy(pad_1d(d["wav"], max_len_wav, constant_values=0))
            for d in datas
        ], dim=0) 

        return basenames, spectrogram, f0, spectrum, aperiodicity, wav