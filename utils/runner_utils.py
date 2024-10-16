import pandas as pd


def load_data(config):
    data_url = config['ehr_dataset_dir']
    if config["mode"] == "test":
        test_pids = pd.read_pickle(f'{data_url}/test_pid.pkl')
        test_y = pd.read_pickle(f'{data_url}/test_y.pkl')
    else:
        test_pids = pd.read_pickle(f'{data_url}/val_pid.pkl')
        test_y = pd.read_pickle(f'{data_url}/val_y.pkl')
    
    return test_pids, test_y


def load_preds(config):
    data_url = config['ehr_dataset_dir']
    doctors = [pd.read_pickle(f'{data_url}/dl_data/{config["ehr_model_names"][i]}_{config["ehr_task"]}_{config["mode"]}_output.pkl') for i in range(config["doctor_num"])]
    return doctors


def check_numbers(config, nums):
    dataset = config['ehr_dataset_name']
    mode = config['mode']
    task = config['ehr_task']
    if dataset in ['mimic-iv', 'mimic-iii']:
        if task == 'outcome':
            if mode == 'test':
                if any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                    return True
            elif mode == 'val':
                if any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                    return True
        elif task == 'readmission':
            if mode == 'test':
                if any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                    return True
            elif mode == 'val':
                if any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                    return True
    elif dataset == 'cdsl':
        if mode == 'test':
            if max(nums) - min(nums) > 0.6 and any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                return True
        elif mode == 'val':
            if max(nums) - min(nums) > 0.6 and any((nums[i] < 0.5 and nums[j] > 0.5) for i in range(len(nums)) for j in range(len(nums))):
                return True
    elif dataset == 'esrd':
        return True
    return False