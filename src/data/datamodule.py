import os
import lightning.pytorch as pl
from torch.utils.data.dataloader import DataLoader

from .dataset import TrackDataset


class Packets(pl.LightningDataModule):
    def __init__(
        self,
        dataset_prepared_path: str,
        batch_size: int,
        datapoint_type: str = "verma",
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__()
        self.dataset_prepared_path = dataset_prepared_path
        self.batch_size = batch_size
        self.datapoint_type = datapoint_type
        self.num_workers = num_workers

    def prepare_data(self):
        file_dirs = sorted(
            [
                os.path.join(self.dataset_prepared_path, file)
                for file in os.listdir(self.dataset_prepared_path)
            ]
        )
        file_dirs = [d for d in file_dirs if os.path.isdir(d)]
        self.train_files, self.valid_files, self.test_files = self.split_data(file_dirs)
        print(self.valid_files)

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train = TrackDataset(
                self.train_files, datapoint_type=self.datapoint_type
            )
            self.val = TrackDataset(
                self.valid_files, datapoint_type=self.datapoint_type
            )
        elif stage == "test":
            self.test = TrackDataset(
                self.test_files, datapoint_type=self.datapoint_type
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def split_data(
        self, file_dirs: list[list[str]]
    ) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
        test_files_len = 1
        valid_files_len = 1

        return (
            file_dirs[: -(test_files_len + valid_files_len)],
            file_dirs[-(valid_files_len + test_files_len) : -test_files_len],
            file_dirs[-test_files_len:],
        )