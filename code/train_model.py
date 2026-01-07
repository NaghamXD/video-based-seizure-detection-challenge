import sys
import os
import glob
import numpy as np
import pandas as pd
import utils


def explore_train_data(train_data_dir='../train_data'):
    if train_data_dir is None:
        return
    
    _npy_paths = glob.glob(train_data_dir + '/*.npy')
    _train_data_df = pd.read_csv(os.path.join(train_data_dir, 'train_data.csv'))
    print('npy paths', len(_npy_paths), 'labels', _train_data_df.shape, _train_data_df['label'].sum())


def train_model(train_data_dir='../train_data', max_elem=100):
    _train_data_df = pd.read_csv(os.path.join(train_data_dir, 'train_data.csv'))
    _means = []
    for i in range(len(_train_data_df)):
        if max_elem is not None and i >= max_elem:
            break

        _name = _train_data_df.iloc[i]['segment_name']
        _label = _train_data_df.iloc[i]['label']
        _lmk_arr = np.load(os.path.join(train_data_dir, _name))

        print(i, _name, _label, _lmk_arr.shape, np.sum(np.isnan(_lmk_arr)))

        # Replace np.nan values
        utils.fill_nan_values(_lmk_arr)

        if _label == 1:
            _means += [np.mean(_lmk_arr) * 2.0]
        else:
            _means += [np.mean(_lmk_arr) * 0.5]

    np.save('model.npy', np.mean(_means))


def get_trained_model():
    _model = np.load('model.npy')
    # print('model', _model, 0.5 > _model)
    return _model


if __name__ == '__main__':
    print(utils.fill_nan_values)
    explore_train_data()
    train_model()
    get_trained_model()
