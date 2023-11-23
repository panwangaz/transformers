import torch
import pandas as pd
from datasets import Dataset
from typing import Iterator, Optional, Generic, Sized, TypeVar

T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, kick_ratio: float = None) -> None:
        self.origin_data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.kick_ratio = kick_ratio

        self.origin_data_source_list = self.origin_data_source.to_list()
        self.copy_group_index_list = []
        self.raw_p_group_index_list = []
        self.raw_m_group_index_list = []
        self.raw_n_group_index_list = []
        error_data_rows = []
        for index, item in enumerate(self.origin_data_source_list):
            if item["group_id"] == 1:
                self.copy_group_index_list.append(index)
            elif item["label"] == 2:
                self.raw_p_group_index_list.append(index)
            elif item["label"] == 1:
                self.raw_m_group_index_list.append(index)
            elif item["label"] == 0:
                self.raw_n_group_index_list.append(index)
            else:
                error_data_rows.append(item)
        self.refresh_data_source()

        if len(error_data_rows) > 0:
            print(error_data_rows)
            raise ValueError("some rows of data error!!!")

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def refresh_data_source(self):
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # kick data, n >> p
        # Dataset.from_pandas(pd.DataFrame(test_dataset_list))
        if self.kick_ratio is not None:
            kick_n_len = int(self.kick_ratio * len(self.copy_group_index_list))
            random_copy_index_list = torch.randperm(len(self.copy_group_index_list), generator=generator).tolist()[kick_n_len:]
            random_m_index_list = torch.randperm(len(self.raw_m_group_index_list), generator=generator).tolist()[kick_n_len:]
            sampled_index_list = []
            sampled_index_list.extend(self.raw_p_group_index_list)
            sampled_index_list.extend(self.raw_n_group_index_list)
            sampled_index_list.extend(random_m_index_list)
            sampled_index_list.extend(random_copy_index_list)
            data_source_list = [self.origin_data_source_list[index] for index in sampled_index_list]
            self.data_source = Dataset.from_pandas(pd.DataFrame(data_source_list))
            print("=="*80)
            print("kick resample done...")
            print("=="*80)
        else:
            self.data_source = self.origin_data_source

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # kick data, n >> p
        # Dataset.from_pandas(pd.DataFrame(test_dataset_list))
        self.refresh_data_source()

        n = len(self.data_source)

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples
