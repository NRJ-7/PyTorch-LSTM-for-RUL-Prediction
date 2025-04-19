from loading_data import *
from model import *
from nrj.model_2 import *
from visualize import *
import numpy as np
from main import *
import torch
import plotly.express as px
import plotly.io as pio
# pio.templates.default = "none"
pio.templates.default = "plotly_dark"

N_HIDDEN = 96  # NUMBER OF HIDDEN STATES
N_LAYER = 4  # NUMBER OF LSTM LAYERS
N_EPOCH = 150  # NUM OF EPOCHS
RUL_UPPER_BOUND = 135  # UPPER BOUND OF RUL
LR = 0.01  # LEARNING RATE

def get_rmse(arr1, arr2):
    """
    Calculate the RMSE between two arrays.
    """
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    rmse = np.sqrt(np.mean((arr1 - arr2) ** 2))
    return rmse

def get_X_y(unit_nr, group, test=False):
    """
    Get the X and y data for a specific unit number and group.
    """
    if test:
        X = group.get_group(unit_nr).iloc[:, 2:]
        X_tensors = Variable(torch.Tensor(X.to_numpy()))
        X_tensors = torch.reshape(X_tensors, (X_tensors.shape[0], 1, X_tensors.shape[1]))
        y_tensors = None
    else:
        X, y = group.get_group(unit_nr).iloc[:, 2:-1], group.get_group(unit_nr).iloc[:, -1:]
        X_tensors = Variable(torch.Tensor(X.to_numpy()))
        y_tensors = Variable(torch.Tensor(y.to_numpy()))
        X_tensors = torch.reshape(X_tensors, (X_tensors.shape[0], 1, X_tensors.shape[1]))
    
    return X_tensors, y_tensors

def plot_rul_vs_time(unit_nr, model, group_test, test=False, y_test_array=None, clip=False):
    """
    Plot the RUL over time for a given unit number.
    """
    if test:
        assert y_test_array is not None, "y_test_array must be provided for test data"

    X_tensors, y_tensors = get_X_y(unit_nr, group_test, test)
    
    model.eval()
    predicted_rul = model(X_tensors).detach().numpy()
    
    if test:
        strt, end = y_test_array[unit_nr - 1] + X_tensors.shape[0] - 1, y_test_array[unit_nr - 1] - 1
        true_rul = np.arange(strt, end, -1).reshape(predicted_rul.shape)
        if clip:
            true_rul.clip(max=RUL_UPPER_BOUND, out=true_rul)
    else:
        true_rul = y_tensors.detach().numpy()

    fig = px.line(
        x=np.arange(1, len(true_rul) + 1),
        y=true_rul.reshape(-1),
        title=f'RUL Over Time: Unit {unit_nr}',
        labels={'x': 'Time', 'y': 'RUL'},
        markers=False
    )
    fig.add_scatter(
        x=np.arange(1, len(predicted_rul) + 1),
        y=predicted_rul.reshape(-1),
        mode='lines',
        name='Predicted RUL'
    )
    fig.update_layout(
        legend=dict(yanchor="bottom",
                    y=0.1,
                    xanchor="left",
                    x=0.1)
    )
    fig.show()
    
def normalize_data(df, type='minmax', test=False, stats = None):
    """
    Normalize the input data.
    """ 
    title = df.iloc[:, 0:2]
    data = df.iloc[:, 2:]
    
    if not test:
        if type == 'minmax':
            train_min, train_max = data.min(), data.max()
            data_norm = (data - train_min) / (train_max - train_min)
            stats = (train_min, train_max)
        elif type == 'zscore':
            train_mean, train_std = data.mean(), data.std()
            data_norm = (data - train_mean) / train_std
            stats = (train_mean, train_std)
        else:
            raise ValueError("Normalization type must be 'minmax' or 'zscore'")
        return pd.concat([title, data_norm], axis=1), stats
    else:
        assert stats is not None, "stats must be provided for test data"
        if type == 'minmax':
            train_min, train_max = stats
            data_norm = (data - train_min) / (train_max - train_min)
        elif type == 'zscore':
            train_mean, train_std = stats
            data_norm = (data - train_mean) / train_std
        else:
            raise ValueError("Normalization type must be 'minmax' or 'zscore'")
        return pd.concat([title, data_norm], axis=1)       
    
def load_FD001(rul_ub=None, norm_type='minmax'):
    """
    Load and preprocess the FD001 dataset.
    :param rul_ub: upper limit for target RULs
    :return: grouped data per sample
    """
    # load data FD001.py
    # define filepath to read data
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train_data = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test_data = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features, derived from EDA
    drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train_data.drop(labels=drop_labels, axis=1, inplace=True)
    test_data.drop(labels=drop_labels, axis=1, inplace=True)
    
    # normalize data
    train_norm, train_stats = normalize_data(train_data, type=norm_type)
    test_norm = normalize_data(test_data, type=norm_type, test=True, stats=train_stats)
    
    # add RUL to training data
    # as in piece-wise linear function, there is an upper limit for target RUL,
    # however, experimental results shows this goes even better without it:
    train_norm = add_rul_1(train_norm)
    if rul_ub is not None:
        train_norm['RUL'].clip(upper=rul_ub, inplace=True)
        
    group_test = test_norm.groupby(by="unit_nr")
    group = train_norm.groupby(by="unit_nr")

    return group, group_test, y_test