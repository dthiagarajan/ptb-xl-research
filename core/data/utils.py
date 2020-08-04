import ast
from functools import partial
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import wfdb


def load_csv(csv_path, index_col=None):
    return pd.read_csv(csv_path, index_col=index_col)


def load_numpy_array(input_fp):
    print(f'Loading Numpy array from .npy at {input_fp}...')
    arr = np.load(input_fp)
    print(f'...done.')
    return arr


def load_df(input_fp):
    print(f'Loading DataFrame from JSON at {input_fp}...')
    df = pd.read_json(input_fp, orient='index')
    print(f'...done.')
    return df


def save_numpy_array(arr, output_fp):
    print(f'Saving Numpy array to .npy at {output_fp}...')
    np.save(output_fp, arr)
    print(f'...done.')


def save_df(df, output_fp):
    print(f'Saving DataFrame to JSON at {output_fp}...')
    df.to_json(output_fp, orient='index')
    print(f'...done.')


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_lr, desc='wfdb.rdsamp')]
    else:
        data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_hr, desc='wfdb.rdsamp')]
    data = np.array([signal for signal, meta in tqdm(data, desc='np.array conversion')])
    return data


def aggregate_diagnostic(record_scp_codes, scp_codes_df=None):
    if scp_codes_df is None:
        raise NotImplementedError(
            'This function must be called after wrapping with functools.partial'
        )
    return set([
        scp_codes_df.loc[key].diagnostic_class for key in record_scp_codes.keys()
        if key in scp_codes_df.index
    ])


def load_all_data(path_to_ptbxl, sampling_rate, checkpoint_path='.', save=False):
    if (
        Path(f'{checkpoint_path}/all_data_{sampling_rate}.npy').exists() and
        Path(f'{checkpoint_path}/all_labels.json').exists()
    ):
        print('Data and labels already exist, loading...')
        X, Y = (
            load_numpy_array(f'{checkpoint_path}/all_data_{sampling_rate}.npy'),
            load_df(f'{checkpoint_path}/all_labels.json')
        )
    else:
        Y = load_csv(os.path.join(path_to_ptbxl, 'ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        X = load_raw_data(Y, sampling_rate, path_to_ptbxl)

        # Load scp_statements.csv for diagnostic aggregation
        scp_codes_df = load_csv(os.path.join(path_to_ptbxl, 'scp_statements.csv'), index_col=0)
        scp_codes_df = scp_codes_df[scp_codes_df.diagnostic == 1]

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(
            partial(aggregate_diagnostic, scp_codes_df=scp_codes_df)
        )

        if save:
            print('Saving data to JSON...')
            save_numpy_array(X, f'./all_data_{sampling_rate}.npy')
            print('Saving labels to JSON...')
            save_df(Y, './all_labels.json')
    return X, Y


def split_all_data(X, Y, test_fold=10):
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    return X_train, y_train, X_test, y_test
