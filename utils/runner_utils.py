import pandas as pd


def load_data(config):
    data_url = config['ehr_dataset_dir']
    train_pids = pd.read_pickle(f'{data_url}/train_pid.pkl')
    val_pids = pd.read_pickle(f'{data_url}/val_pid.pkl')
    if config["mode"] == "test":
        test_pids = pd.read_pickle(f'{data_url}/test_pid.pkl')
    else:
        test_pids = pd.read_pickle(f'{data_url}/val_pid.pkl')

    train_y = pd.read_pickle(f'{data_url}/train_y.pkl')
    test_y = pd.read_pickle(f'{data_url}/test_y.pkl')
    val_y = pd.read_pickle(f'{data_url}/val_y.pkl')
    
    return train_pids,test_pids,val_pids,train_y,test_y,val_y


def load_preds(config):
    data_url = config['ehr_dataset_dir']
    if config["mode"] == "test":
        doctors = [pd.read_pickle(f'{data_url}/{config["ehr_model_names"][i]}_seed{config["seeds"][i]}_output.pkl') for i in range(config["doctor_num"])]
    else:
        doctors = [pd.read_pickle(f'{data_url}/{config["ehr_model_names"][i]}_seed{config["seeds"][i]}_output2.pkl') for i in range(config["doctor_num"])]
    return doctors


def check_numbers(dataset, mode, nums):
    if dataset in ['mimic-iv', 'mimic-iii']:
        if mode == 'test':
            if max(nums) - min(nums) > 0.4 or any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                return True
        elif mode == 'val':
            if max(nums) - min(nums) > 0.4 or any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                return True
    elif dataset == 'cdsl':
        if mode == 'test':
            if max(nums) - min(nums) > 0.4 and any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                return True
        elif mode == 'val':
            if max(nums) - min(nums) > 0.4 or any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                return True
    return False