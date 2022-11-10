import itertools

import numpy as np


class MultiLoader:
    """Helper for loading data from a list of dataloaders."""

    def __init__(self, loaders):
        self.loaders = loaders
        self.dataset_lens = [len(loader.dataset) for loader in loaders]
        self.bins = np.cumsum(self.dataset_lens).tolist()
        assert len(self.loaders) == len(self.dataset_lens)
        # self.dataset = self

    def __getitem__(self, idx):
        # Which loader should we look at?
        dataset_idx = np.digitize(idx, self.bins)
        assert dataset_idx < len(self.loaders), f"dataset_idx is {dataset_idx}, idx is {idx}"
        idx_to_subtract = self.bins[dataset_idx]

        idx -= idx_to_subtract
        if idx > len(self):
            raise IndexError(f"idx is {idx}, len is {len(self)}")

        return self.loaders[dataset_idx].dataset[idx]

    def __len__(self):
        return sum(self.dataset_lens)

    def __iter__(self):
        return itertools.chain(*self.loaders)
