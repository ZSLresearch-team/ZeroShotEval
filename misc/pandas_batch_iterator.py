import pandas as pd

from math import ceil


class PandasBatchIterator:
    """Iterates pandas dataframe in batches"""

    def __init__(self, dataframe, batch_size=64, columns=None):
        """
        Args:
            dataframe (pandas.DataFrame): input dataframe
            batch_size (int): batch size
            columns (list): columns to include in output dataframe
                (returns all by default)
        """
        self._dataframe = dataframe
        self._batch_size = batch_size
        self._columns = columns

        self._len = ceil(self._dataframe.shape[0] / self._batch_size)

        self._run_tests()

    def _run_tests(self):
        assert self._batch_size > 0

        if self._columns:
            for column in self._columns:
                assert column in self._dataframe.columns

    def reset(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def next(self):
        return next(self)

    def __len__(self):
        return self._len

    def __next__(self):
        for cur_batch in range(self._len):
            cur_idx = cur_batch * self._batch_size
            end_idx = cur_idx + self._batch_size - 1

            if self._columns:
                batch = self._dataframe.loc[cur_idx:end_idx, self._columns]
            else:
                batch = self._dataframe.loc[cur_idx:end_idx]

            yield batch


# #%%
# import pandas as pd
# import numpy as np
# #%%
# df = pd.DataFrame(np.random.randn(68, 4)).loc[:, :]

# #%%
# it = PandasBatchIterator(df, columns=[0, 3])
# for el in it:
#     print(el)
#     print('----------------------')
# #%%


# #%%
# for el in range(5):
#     assert el in df.columns

# #%%
