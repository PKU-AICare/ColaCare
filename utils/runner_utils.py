import torch
import torch.nn as nn
import pandas as pd

def get_loss(pred, truth) -> torch.Tensor:
    return nn.BCELoss()(pred, truth)


def load_data(data_url):
    train_pids = pd.read_pickle(f'{data_url}/train_pid.pkl')
    test_pids = pd.read_pickle(f'{data_url}/test_pid.pkl')
    val_pids = pd.read_pickle(f'{data_url}/val_pid.pkl')

    train_y = pd.read_pickle(f'{data_url}/train_y.pkl')
    test_y = pd.read_pickle(f'{data_url}/test_y.pkl')
    val_y = pd.read_pickle(f'{data_url}/val_y.pkl')
    
    return train_pids,test_pids,val_pids,train_y,test_y,val_y


def load_all_data(data_url):
    train_pids = pd.read_pickle(f'{data_url}/train_pid.pkl')
    test_pids = pd.read_pickle(f'{data_url}/test_pid.pkl')
    val_pids = pd.read_pickle(f'{data_url}/val_pid.pkl')
    
    all_pids = train_pids + test_pids + val_pids
    return all_pids, train_pids, test_pids, val_pids


def load_preds(config):
    data_url = config['ehr_dataset_dir']
    doctors = [pd.read_pickle(f'{data_url}/{config["ehr_model_names"][i]}_seed{config["seeds"][i]}_output.pkl') for i in range(config["doctor_num"])]
    
    return doctors


def check_numbers(nums):
    if max(nums) - min(nums) > 0.2:
        return True
    
    if any((nums[i] < 0.5 and nums[j] > 0.5) or (nums[i] > 0.5 and nums[j] < 0.5) 
           for i in range(len(nums)) for j in range(len(nums))):
        return True
    
    return False