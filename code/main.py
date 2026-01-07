import sys
import os
import glob
import numpy as np
import pandas as pd
import train_model
import utils


if __name__ == '__main__':
    _test_data_dir = sys.argv[1]
    _output_data_dir = sys.argv[2]
    print(sys.argv)

    _test_data_df = pd.DataFrame(columns=['segment_name', 'label'])
    _segment_names, _labels = [], []
    
    # Get the pre-trained model to use in inference
    _model = train_model.get_trained_model()
    print('Trained model loaded:', _model)

    # Read all .npy file paths from the given test_dir
    _paths = glob.glob(_test_data_dir + '/*.npy')
    print('The found .npy landmarks files:', len(_paths))

    # Iterate over each path to load the landmarks array with shape (150, 33, 5)
    for i in range(len(_paths)):
        _p = _paths[i]
        _name = os.path.basename(_p)
        _lmk_arr = np.load(_p)

        # Replace nan values
        utils.fill_nan_values(_lmk_arr)

        # Do inferencing
        _mean = np.mean(_lmk_arr)
        if _mean > _model:
            _label = 1
        else:
            _label = 0
        _segment_names += [_name]
        _labels += [_label]
        # print(i, _p, _name, _label, _lmk_arr.shape, _mean, _model)

    _test_data_df['segment_name'] = _segment_names
    _test_data_df['label'] = _labels

    # Save predictions into a single csv file inside output_dir
    _test_data_df.to_csv(os.path.join(_output_data_dir, 'test_data.csv'), sep=',', index=False)
    print('Inference done, saving csv file: test_data.csv', _test_data_df.shape) 
