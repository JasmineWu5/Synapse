import lightning as L
import torch

from core.config import DataConfig, RunConfig
from core.dataset import MapStyleDataset #, HybridDataset

class DataModule(L.LightningDataModule):
    def __init__(
            self,
            data_cfg: DataConfig,
            run_cfg: RunConfig,
            train_file_list: list[str] | None = None,
            val_file_list: list[str] | None = None,
            test_file_list: list[str] | None = None,
    ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_file_list = train_file_list
        self.val_file_list = val_file_list
        self.test_file_list = test_file_list
        self.data_cfg = data_cfg
        self.run_cfg = run_cfg

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MapStyleDataset(
                self.train_file_list,
                'train',
                self.data_cfg
            )
            self.val_dataset = MapStyleDataset(
                self.val_file_list,
                'val',
                self.data_cfg
            )
        if stage == "test" or stage is None:
            self.test_dataset = MapStyleDataset(
                self.test_file_list,
                'test',
                self.data_cfg
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.run_cfg.batch_size,
            num_workers=min(self.run_cfg.num_workers, len(self.train_dataset)),
            pin_memory=True,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.run_cfg.batch_size,
            num_workers=min(self.run_cfg.num_workers, len(self.train_dataset)),
            pin_memory=True,
            persistent_workers=True,
            shuffle = False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.run_cfg.batch_size,
            num_workers=min(self.run_cfg.num_workers, len(self.train_dataset)),
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )