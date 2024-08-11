import os
from typing import List, Literal

import pandas as pd
import numpy as np

from .datasets_info import *


def get_data_from_files(data_url: str, model: str, seed: int, mode: str, patient_id, patient_index: int=0):
    if mode == 'test':
        x = pd.read_pickle(os.path.join(data_url, "test_x.pkl"))[patient_index]
        raw_x = pd.read_pickle(os.path.join(data_url, "test_raw_x.pkl"))[patient_index]
        y = pd.read_pickle(os.path.join(data_url, f"{model}_seed{seed}_output.pkl"))[patient_index]
        important_features = pd.read_pickle(os.path.join(data_url, f"{model}_seed{seed}_features.pkl"))[patient_index]
    elif mode == 'val':
        x = pd.read_pickle(os.path.join(data_url, "sub_test_x.pkl"))[patient_index]
        raw_x = pd.read_pickle(os.path.join(data_url, "sub_test_raw_x.pkl"))[patient_index]
        y = pd.read_pickle(os.path.join(data_url, f"{model}_seed{seed}_output2.pkl"))[patient_index]
        important_features = pd.read_pickle(os.path.join(data_url, f"{model}_seed{seed}_features2.pkl"))[patient_index]
    features = pd.read_pickle(os.path.join(data_url, "labtest_features.pkl"))
    return x, raw_x, features, y, important_features


def get_var_desc(var: float):
    if var > 0:
        return round(var * 100, 2)
    else:
        return round(-var * 100, 2)


def get_trend_desc(var: float):
    if var > 0:
        return "increased"
    else:
        return "decreased"
    
    
def get_recommended_trend_desc(var: float):
    if var > 0:
        return "decrease"
    else:
        return "increase"
    
    
def get_range_desc(key: str, var: float):
    if key in ["Weight", "Appetite"]:
        return ""
    if var < medical_standard[key][0]:
        return f"the value is lower than normal range by {round((medical_standard[key][0] - var) / medical_standard[key][0] * 100, 2)}%"
    elif var > medical_standard[key][1]:
        return f"the value is higher than normal range by {round((var - medical_standard[key][1]) / medical_standard[key][1] * 100, 2)}%"
    else:
        return "the value is within the normal range"


def get_mean_desc(var: str, mean: float):
    if var < mean:
        return f"{round((mean - var) / mean * 100, 0)}% lower"
    elif var > mean:
        return f"{round((var - mean) / mean * 100, 0)}% higher"


def get_death_desc(risk: float):
    if risk < 0.5:
        return "a low level"
    elif risk < 0.7:
        return "a high level"
    else:
        return "an extremely high level"
    
    
def get_distribution(data, values):
    arr = np.sort(np.array(values))
    index = np.searchsorted(arr, data, side='right')
    rank = index / len(arr) * 100
    if rank < 40:
        return "at the bottom 40%"
    elif rank < 70:
        return "at the middle 30%"
    else:
        return "at the top 30%"


def format_input_ehr(raw_x: List[List[float]], features: List[str]):
    ehr = ""
    for i, feature in enumerate(features):
        name = medical_name[feature] if feature in medical_name else feature
        ehr += f"- {name}: \"{', '.join(list(map(lambda x: str(round(x, 2)), raw_x[:, i])))}\"\n"
    return ehr


def generate_prompt(dataset: str, data_url: str, patient_index: int, patient_id: int):
    if dataset == 'esrd':
        basic_data = pd.read_pickle(os.path.join(data_url, 'basic.pkl'))[patient_id]
        gender = "male" if basic_data["Gender"] == 1 else "female"
        age = basic_data["Age"]
        if " " in basic_data["Origin_disease"]:
            ori_disease = basic_data["Origin_disease"].split(" ")[0]
            ori_disease = original_disease[ori_disease]
        else:
            ori_disease = original_disease[basic_data["Origin_disease"]]
        basic_disease = [disease_english[key] for key in disease_english.keys() if basic_data[key] == 1]
        basic_disease = ", and basic disease " + ", ".join(basic_disease) if len(basic_disease) > 0 else ""
        basic_context = f"This {gender} patient, aged {age}, is an End-Stage Renal Disease(ESRD) patient with original disease {ori_disease}{basic_disease}.\n"
    elif dataset == 'cdsl':
        basic_data = pd.read_pickle(os.path.join(data_url, 'basic.pkl'))[patient_id]
        gender = "male" if basic_data["Sex"] == 1 else "female"
        age = basic_data["Age"]
        basic_context = f"This {gender} patient, aged {age}, is an patient admitted with a diagnosis of COVID-19 or suspected COVID-19 infection.\n"
    else: # [mimic-iii, mimic-iv]
        basic_context = '\n'

    models = ['LR', 'ConCare']
    last_visit_context = f"We have {len(models)} models {', '.join(models)} to predict the mortality risk and estimate the feature importance weight for the patient in the last visit:\n"
    for model in models:
        _, raw_x, features, y, important_features = get_data_from_files(data_url, model, patient_index)
        ehr_context = "Here is complete medical information from multiple visits of a patient, with each feature within this data as a string of values separated by commas.\n" + format_input_ehr(raw_x, features)

        last_visit = f"The mortality prediction risk for the patient from {model} model is {round(float(y), 2)} out of 1.0, which means the patient is at {get_death_desc(float(y))} of death risk. Our model especially pays great attention to following features:\n"

        survival_stats = pd.read_pickle(os.path.join(data_url, 'survival.pkl'))
        dead_stats = pd.read_pickle(os.path.join(data_url, 'dead.pkl'))
        for item in important_features:
            key, value = item
            if key in ['Weight', 'Appetite']:
                continue
            survival_mean = survival_stats[key]['mean']
            dead_mean = dead_stats[key]['mean']
            key_name = medical_name[key] if key in medical_name else key
            key_unit = ' ' + medical_unit[key] if key in medical_unit else ''
            last_visit += f'{key_name}: with '
            if model == 'ConCare':
                last_visit += f'importance weight of {round(float(value["attention"]), 3)} out of 1.0. '
            else:
                last_visit += f'shap value of {round(float(value["attention"]), 3)}. '
            last_visit += f'The feature value is {round(value["value"], 2)}{key_unit}, which is {get_mean_desc(value["value"], survival_mean)} than the average value of survival patients ({round(survival_mean, 2)}{key_unit}), {get_mean_desc(value["value"], dead_mean)} than the average value of dead patients ({round(dead_mean, 2)}{key_unit}).\n'
        last_visit_context += last_visit + '\n'
    
    # similar_patients = get_similar_patients(data_url, patient_id)
    # similar_context = "The AI model has found similar patients to the patient, including:\n"
    # for idx, patient in enumerate(similar_patients):
    #     similar_context += f"Patient {idx + 1}: {patient['gender']}, {patient['age']} years old, with original disease {patient['oriDisease']}{patient['basicDisease']}{patient['deathText']}.\n"


    subcontext = basic_context + last_visit_context
    hcontext = basic_context + '\n' + ehr_context + '\n' + last_visit_context

    return subcontext, hcontext


class ContextBuilder:
    def __init__(self, dataset_name: str, model_name: str, seed: int, mode: str) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = f"ehr_datasets/{dataset_name}/processed/fold_1"
        self.model_name = model_name
        self.seed = seed
        self.mode = mode

    def generate_context(self, patient_index: int, patient_id):
        """Generate healthcare context for the patient.

        Args:
            patient_index (int): The index of the patient in the dataset.
            patient_id (str | int): The ID of the patient in the dataset.

        Returns:
            List[str]: The healthcare context for the patient.
        """
        if self.dataset_name == 'esrd':
            basic_data = pd.read_pickle(os.path.join(self.dataset_dir, 'basic.pkl'))[patient_id]
            gender = "male" if basic_data["Gender"] == 1 else "female"
            age = basic_data["Age"]
            if " " in basic_data["Origin_disease"]:
                ori_disease = basic_data["Origin_disease"].split(" ")[0]
                ori_disease = original_disease[ori_disease]
            else:
                ori_disease = original_disease[basic_data["Origin_disease"]]
            basic_disease = [disease_english[key] for key in disease_english.keys() if basic_data[key] == 1]
            basic_disease = ", and basic disease " + ", ".join(basic_disease) if len(basic_disease) > 0 else ""
            basic_context = f"This {gender} patient, aged {age}, is an End-Stage Renal Disease(ESRD) patient with original disease {ori_disease}{basic_disease}.\n"
        elif self.dataset_name == 'cdsl':
            basic_data = pd.read_pickle(os.path.join(self.dataset_dir, 'basic.pkl'))[patient_id]
            gender = "male" if basic_data["Sex"] == 1 else "female"
            age = basic_data["Age"]
            basic_context = f"This {gender} patient, aged {age}, is an patient admitted with a diagnosis of COVID-19 or suspected COVID-19 infection.\n"
        else: # [mimic-iii, mimic-iv]
            basic_data = pd.read_pickle(os.path.join(self.dataset_dir, 'basic.pkl'))[patient_id]
            gender = "male" if basic_data["Sex"] == 1 else "female"
            age = basic_data["Age"]
            basic_context = f"This {gender} patient, aged {age}, is an patient in Intensive Care Unit (ICU).\n"

        _, raw_x, features, y, important_features = get_data_from_files(self.dataset_dir, self.model_name, self.seed, self.mode, patient_id, patient_index)
        
        if self.dataset_name in ['mimic-iv', 'mimic-iii']:
            raw_x = np.array(raw_x)[:, 2 + 47:]
        elif self.dataset_name == 'esrd':
            raw_x = np.array(raw_x)[:, 4:]
        else:
            raw_x =  np.array(raw_x)[:, 2:]
        ehr_context = "Here is complete medical information from multiple visits of a patient, with each feature within this data as a string of values separated by commas.\n" + format_input_ehr(raw_x, features)

        last_visit = f"The mortality prediction risk for the patient from {self.model_name} model is {round(float(y), 2)} out of 1.0, which means the patient is at {get_death_desc(float(y))} of death risk. Our model especially pays great attention to following features:\n"

        survival_stats = pd.read_pickle(os.path.join(self.dataset_dir, 'survival.pkl'))
        dead_stats = pd.read_pickle(os.path.join(self.dataset_dir, 'dead.pkl'))
        for item in important_features:
            key, value = item
            if key in ['Weight', 'Appetite']:
                continue
            survival_mean = survival_stats[key]['mean']
            dead_mean = dead_stats[key]['mean']
            key_name = medical_name[key] if key in medical_name else key
            if self.dataset_name == 'esrd':            
                key_unit = ' ' + medical_unit[key] if key in medical_unit else ''
            elif self.dataset_name == 'mimic-iv':
                key_unit = ' ' + mimic_unit[key] if key in mimic_unit else ''
            else:
                key_unit = ''
            last_visit += f'{key_name}: with '
            if self.model_name == 'ConCare':
                last_visit += f'importance weight of {round(float(value["attention"]), 3)} out of 1.0. '
            else:
                last_visit += f'shap value of {round(float(value["attention"]), 3)}. '
            last_visit += f'The feature value is {round(value["value"], 2)}{key_unit}, which is {get_mean_desc(value["value"], survival_mean)} than the average value of survival patients ({round(survival_mean, 2)}{key_unit}), {get_mean_desc(value["value"], dead_mean)} than the average value of deceased patients ({round(dead_mean, 2)}{key_unit}).\n'
            if self.dataset_name == 'esrd':            
                last_visit += f" The reference range is {medical_standard[key][0]}{key_unit} to {medical_standard[key][1]}{key_unit}.\n" if key in medical_standard else ""
            elif self.dataset_name == 'mimic-iv':
                last_visit += f" The reference range is {mimic_range[key]}.\n" if key in mimic_range else ""
            else:
                pass
        last_visit_context = last_visit + '\n'

        subcontext = basic_context + last_visit_context
        hcontext = basic_context + '\n' + ehr_context + '\n' + last_visit_context

        return basic_context, subcontext, hcontext