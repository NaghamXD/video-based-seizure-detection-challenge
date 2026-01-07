import sys
import os
import numpy as np


def fill_nan_values(lmk_arr, fill_value=-1):
    if lmk_arr is None:
        return 

    _nan_inx = np.isnan(lmk_arr)
    lmk_arr[_nan_inx] = -1
    return lmk_arr


if __name__ == '__main__':
    a = np.random.rand(150, 33, 5)
    a[0] = np.nan
    a[:, 1, :] = np.nan
    a[:, :, 3] = np.nan
    print(np.sum(np.isnan(a)))
    fill_nan_values(lmk_arr=a)
    print(np.sum(np.isnan(a)), np.sum(a == -1))
