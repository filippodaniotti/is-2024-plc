import os
import torch
import numpy as np
from torch import tensor
from torch.utils.data import Dataset


def _load_datapoint_parcnet(packet_file: str):
    datapoint = np.load(packet_file)
    lost_packet = tensor(datapoint["lost_packet"]).float()
    last_packet = tensor(datapoint["last_packet"]).float()
    linear_prediction = tensor(datapoint["linear_prediction"]).float()
    start_sample_idx = tensor(datapoint["start_sample_idx"])
    return (
        lost_packet,
        last_packet,
        linear_prediction,
        start_sample_idx,
    )


class TrackDataset(Dataset):
    def __init__(self, data_dirs: list[list[str]], datapoint_type: str) -> None:
        self.packet_files = self._unpack_files(data_dirs)
        self.data_dirs = data_dirs

        self._load_datapoint = _load_datapoint_parcnet

    def __len__(self) -> int:
        return len(self.packet_files)

    def __getitem__(self, index: int):
        return self._load_datapoint(self.packet_files[index])

    def _unpack_files(self, data_dirs: list[list[str]]) -> list[str]:
        return [
            f"{os.path.join(file_dir, f)}"
            for file_dir in data_dirs if os.path.isdir(file_dir)
            for f in os.listdir(file_dir)
        ]