import torch
import collections.abc

from typing import Sequence, Callable, Any
from torch.utils.data import Dataset, Subset
from monai.transforms import apply_transform
from monai.transforms import Compose
from .utils import datafold_read

class BraTs21_Dataset(Dataset):
    def __init__(
        self, 
        data_dir: str = '/data/hpc/dqm/data/brats-2021', 
        data_list: list = [],
        transform: Callable | None = None
    ) -> None:
        """
        Args:
            data_dir: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data_dir = data_dir
        self.data_list = data_list
        try:
            self.transform = Compose(transform) if not isinstance(transform, Compose) else transform
        except Exception as e:
            raise ValueError("`transform` must be a callable or a list of callables that is Composable") from e
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data_list[index]
        # return apply_transform(self.transform, data_i) if self.transform is not None else data_i
        return self.transform(data_i)
    

    def __getitem__(self, index: int | slice | Sequence[int]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)
    