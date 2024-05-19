import json
import pdb
import os
import requests
from typing import List, Dict

import torch
import pandas as pd
import numpy as np

from .ckd_info import medical_standard, disease_english, medical_name, medical_unit, original_disease
from .model_utils import run_concare, run_ml_models, get_data_from_files, get_similar_patients


# def get_ehr_summary(config: dict, data_url: str, patient_id: int):
#     x, _, time, _, _, features = get_data_from_files(data_url, patient_id)
#     x = torch.Tensor(x) # [ts, d + f]
#     x = x[:, config["demo_dim"]:] # [ts, f]
#     y, _, feat_attn = run_concare(x)
#     time = pd.to_datetime(time).strftime("%Y-%m-%d")
    
#     return {"data": summarize_ehr(features, x.detach().numpy().tolist(), feat_attn.tolist(), y.tolist(), time)}

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
        return f"lower than the average value by {round((mean - var) / mean * 100, 0)}%"
    elif var > mean:
        return f"higher than the average value by {round((var - mean) / mean * 100, 0)}%"


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


def generate_prompt(data_url: str, patient_id: int):
    x, raw_x, _, features = get_data_from_files(data_url, patient_id)
    raw_x = np.array(raw_x)
    prompt = ""
    # summarize_result = summarize_ehr(features, raw_x.transpose().tolist(), feat_attn.transpose().tolist(), y.transpose().tolist(), date)

    basic_data = requests.get(f"http://47.93.42.104:10408/v1/app/patients/basics/{patient_id}").json()["data"]
    gender = "male" if basic_data["gender"] == 1 else "female"
    age = basic_data["age"]
    ori_disease = original_disease[basic_data["originDisease"]]
    basic_disease = [disease_english[key] for key in disease_english.keys() if basic_data[key] == 1]
    basic_disease = ", and basic disease " + ", ".join(basic_disease) if len(basic_disease) > 0 else ""
    basic_prompt = f"This {gender} patient, aged {age}, is an End-Stage Renal Disease(ESRD) patient with originial disease {ori_disease}{basic_disease}."
    prompt += basic_prompt + '\n\n'
    # keywords = generate_keywords(summarize_result)
    # keywords_prompt = basic_prompt + ". The AI model pays great attention to the following features: " + ", ".join(keywords) + ".\n"

    similar_patients = get_similar_patients(data_url, patient_id)
    similar_prompt = "The AI model has found similar patients to the patient, including: "
    for idx, patient in enumerate(similar_patients):
        similar_prompt += f"Patient {idx} (similarity: {round(patient['similarity'] * 100, 2)}%, {patient['gender']}, {patient['age']} years old, with originial disease{patient['oriDisease']}{patient['basicDisease']}).\n"

    dl_models = ['ConCare']
    ml_models = ['XGBoost']
    for model in dl_models + ml_models:
        if model in dl_models:
            x = torch.Tensor(x) # [ts, f]
            y, _, feat_attn = run_concare(x)
            feat_attn = feat_attn[-1, :].tolist()
        else:
            config = {'task': 'outcome', 'seed': 42, 'model': model, 'fold': 0}
            x = np.array(x)
            y, _, feat_attn = run_ml_models(config, x)

        last_visit = f"The mortality prediction risk for the patient from {model} model is {round(float(y[-1]) * 100, 2)}, which means the patient is at {get_death_desc(y[-1])} of death risk. Our model pays great attention to following features:\n"

        last_feat_dict = {key: {'value': value, 'attention': attn} for key, value, attn in zip(features, raw_x[-1, :], feat_attn)}
        last_feat_dict_sort = dict(sorted(last_feat_dict.items(), key=lambda x: x[1]['attention'], reverse=True))
        
        idx = 0
        top = 5
        all_df = pd.read_csv(os.path.join(data_url, 'data.csv'))
        survival_stats = pd.read_pickle(os.path.join(data_url, 'survival.pkl'))
        dead_stats = pd.read_pickle(os.path.join(data_url, 'dead.pkl'))
        for key, value in last_feat_dict_sort.items():
            if key in ['Weight', 'Appetite']:
                continue
            if value["attention"] < 0.1:
                break
            survival_mean = survival_stats[key]['mean']
            dead_mean = dead_stats[key]['mean']
            values = all_df[key].values.tolist()
            last_visit += f'{medical_name[key]}: {round(value["value"], 2)} {medical_unit[key]}, with attention of {get_var_desc(value["attention"])}. Among all the ESRD patients, the value is {get_distribution(value["value"], values)}. The average value is {round(survival_mean, 2)} {medical_unit[key]} for survival patients ({get_mean_desc(value["value"], survival_mean)}), {round(dead_mean, 2)} {medical_unit[key]} for dead patients ({get_mean_desc(value["value"], dead_mean)}). According to medical experts, {get_range_desc(key, value["value"])}.\n'
            idx += 1
            if idx >= top:
                break
        prompt += last_visit + '\n'
    
    # last_attention = {key: value for key, value in zip(features, feat_attn[-1, :].tolist())}
    # top2_attention = sorted(last_attention.items(), key=lambda x: x[1], reverse=True)[:2]
    # conclusion = f"The mortality prediction risk in the last visit for the patient from our model is {round(float(y[-1]) * 100, 2)}, which means the patient is at {get_death_desc(y[-1])} of death risk. Our model pays great attention to the two features: {medical_name[top2_attention[0][0]]} and {medical_name[top2_attention[1][0]]}, with the feature attention of {get_var_desc(top2_attention[0][1])}% and {get_var_desc(top2_attention[1][1])}%.\n"

    return prompt


def generate_summary_text(data: List[dict]):
    result = []
    for item in data:
        prompt = ""
        prompt += f'In the period from visit {item["date_index"][0]} to {item["date_index"][1]}:\n'
        if item["type"] == 0:
            prompt += f'The mortality risk fluctuated, rising {get_var_desc(item["risk_variation_absolute"])} to {get_var_desc(item["risk"])} ({get_trend_desc(item["risk_variation"])} {get_var_desc(item["risk_variation"])}%, {get_death_desc(item["risk"])} of death risk).\n'
            prompt += "AI suggests paying attention to: \n"
            for i, lab in enumerate(item["item"]):
                prompt += f'{medical_name[lab]}: {get_trend_desc(item["lab_variation"][i])} from {round(item["lab"][i] - item["lab_variation_absolute"][i], 2)} {medical_unit[lab]} to {round(item["lab"][i], 2)} {medical_unit[lab]} ({get_trend_desc(item["lab_variation"][i])} {get_var_desc(item["lab_variation"][i])}%, {get_range_desc(lab, item["lab"][i])}), with the attention of {get_var_desc(item["lab_attention"][i])}%.\n'
        elif item["type"] == 1:
            prompt += f'Death risk continues to rise {get_var_desc(item["risk_variation_absolute"])} to {get_var_desc(item["risk"])} ({get_trend_desc(item["risk_variation"])} {get_var_desc(item["risk_variation"])}%, {get_death_desc(item["risk"])} of death risk).\n'
            prompt += "AI suggests paying attention to: \n"
            for i, lab in enumerate(item["item"]):
                prompt += f'{medical_name[lab]}: {get_trend_desc(item["lab_variation"][i])} from {round(item["lab"][i] - item["lab_variation_absolute"][i], 2)} {medical_unit[lab]} to {round(item["lab"][i], 2)} {medical_unit[lab]} ({get_trend_desc(item["lab_variation"][i])} {get_var_desc(item["lab_variation"][i])}%, {get_range_desc(lab, item["lab"][i])}), with the attention of {get_var_desc(item["lab_attention"][i])}%.\n'
        elif item["type"] == 2:
            prompt += f"AI continues to focus on "
            prompt += ", ".join(list(map(lambda x: f'{medical_name[x]} (attention of {get_var_desc(item["lab_attention"][item["item"].index(x)])}%)', item["item"]))) + ".\n"
            prompt += "It is recommended to "
            for i, lab in enumerate(item["item"]):
                prompt += f'{get_recommended_trend_desc(item["trend"][i])} {medical_name[lab]} (average of {round(item["lab_avg"][i], 2)}, {get_range_desc(lab, item["lab_avg"][i])}).\n'
        elif item["type"] == 3:
            prompt += "The mortality risk continues to remain high (above 50) during this time period and it is recommended to focus on "
            prompt += ", ".join(list(map(lambda x: f'{medical_name[x]} (attention of {get_var_desc(item["lab_attention"][item["item"].index(x)])}%)', item["item"]))) + ".\n"
        result.append(prompt)
    return result


def generate_keywords(data: List[dict]):
    result = []
    for item in data:
        for lab in item["item"]:
            result.append(medical_name[lab])
    result = list(set(result))
    return result


def summarize_ehr(
    lab_names: List[str],
    lab_values: List[List[float]],
    lab_attention: List[List[float]],
    risk_values: List[float],
    dates: List[str],
):
    flag = False
    flag1 = False
    index = []
    index1 = []
    result = []

    lab_values = {name: value for name, value in zip(lab_names, lab_values)}
    lab_attention = {name: value for name, value in zip(lab_names, lab_attention)}
    length = len(risk_values)

    for i in range(2, length):
        if (
            i < length - 2
            and risk_values[i - 1] >= risk_values[i - 2] * 1.2
            and risk_values[i] >= risk_values[i - 1] * 1.1
            and risk_values[i] >= 0.15
        ):
            flag1 = True
            index1.append([i - 2, i])
        elif (not flag1 or index1[0][1] != i - 1) and risk_values[i - 1] - risk_values[
            i - 2
        ] >= 0.1:
            flag = True
            index.append(i - 1)
    if (not flag1 or index1[0][1] != length - 1) and risk_values[
        length - 1
    ] - risk_values[length - 2] >= 0.1:
        flag = True
        index.append(length - 1)

    if flag:
        index_item = {}
        for item in lab_names:
            for i, ind in enumerate(index):
                lab_variation = lab_values[item][ind] / lab_values[item][ind - 1] - 1
                lab_attention_ = max([lab_attention[item][ind], lab_attention[item][ind - 1]])
                lab_attention_variation = (
                    (lab_attention[item][ind] + 1e-5) / (lab_attention[item][ind - 1] + 1e-5) - 1
                )
            if lab_variation and (
                (
                    lab_attention[item][ind] - lab_attention[item][ind - 1] >= 0.05
                    and lab_attention[item][ind] >= 0.1
                )
                or lab_attention[item][ind] >= 0.25
                or (
                    abs(lab_variation) >= 2
                    and (
                        lab_values[item][ind] > medical_standard[item][1] * 1.2
                        or lab_values[item][ind] < medical_standard[item][0] * 0.8
                    )
                )
            ):
                if index_item.get(dates[ind]):
                    index_item[dates[ind]].append(
                        {
                            "item": item,
                            "date_index": [
                                ind - 1 if ind - 1 > 0 else 0,
                                ind,
                            ],
                            "risk": risk_values[ind],
                            "risk_variation_absolute": risk_values[ind]
                            - risk_values[ind - 1],
                            "risk_variation": risk_values[ind] / risk_values[ind - 1]
                            - 1,
                            "lab": lab_values[item][ind],
                            "lab_variation": lab_variation,
                            "lab_variation_absolute": lab_values[item][ind]
                            - lab_values[item][ind - 1],
                            "lab_attention": lab_attention_,
                            "lab_attention_variation": lab_attention_variation,
                        }
                    )
                else:
                    index_item[dates[ind]] = [
                        {
                            "item": item,
                            "date_index": [ind - 1 if ind - 1 > 0 else 0, ind],
                            "risk": risk_values[ind],
                            "risk_variation_absolute": risk_values[ind]
                            - risk_values[ind - 1],
                            "risk_variation": risk_values[ind] / risk_values[ind - 1]
                            - 1,
                            "lab": lab_values[item][ind],
                            "lab_variation": lab_variation,
                            "lab_variation_absolute": lab_values[item][ind]
                            - lab_values[item][ind - 1],
                            "lab_attention": lab_attention_,
                            "lab_attention_variation": lab_attention_variation,
                        }
                    ]

        for date, items in index_item.items():
            items = sorted(items, key=lambda x: x["lab_attention"], reverse=True)[:3]
            result.append(
                {
                    "type": 0,
                    "date": date[:10],
                    "merge": items[0]["date_index"][0] == length - 1,
                    "date_index": items[0]["date_index"],
                    "item": [item["item"] for item in items],
                    "risk": items[0]["risk"],
                    "risk_variation_absolute": items[0]["risk_variation_absolute"],
                    "risk_variation": items[0]["risk_variation"],
                    "lab": [item["lab"] for item in items],
                    "lab_variation": [item["lab_variation"] for item in items],
                    "lab_variation_absolute": [
                        item["lab_variation_absolute"] for item in items
                    ],
                    "lab_attention": [
                        item["lab_attention"] for item in items
                    ],
                    "lab_attention_variation": [
                        item["lab_attention_variation"] for item in items
                    ],
                }
            )

    if flag1:
        index1_merge = []
        left, right = -1, -1
        for i, ind in enumerate(index1):
            if ind[0] > right:
                if left >= 0:
                    index1_merge.append([left, right])
                left = ind[0]
            right = ind[1]
        index1_merge.append([left, right])
        index_item = {}
        for item in lab_names:
            for i, ind in enumerate(index1_merge):
                lab_attention_ = sum(lab_attention[item][ind[0]: ind[1] + 1]) / (
                    ind[1] - ind[0] + 1
                )
                lab_attention_variation = (lab_attention[item][ind[1]] + 1e-5) / (
                    lab_attention[item][ind[0]] + 1e-5
                ) - 1
                lab_variation = lab_values[item][ind[1]] / lab_values[item][ind[0]] - 1
            if (
                (lab_attention_variation >= 0.2 and lab_attention[item][ind[1]] >= 0.05)
                or lab_attention[item][ind[1]] >= 0.15
                or (
                    abs(lab_variation) >= 2
                    and (
                        lab_values[item][ind[1]] > medical_standard[item][1] * 1.2
                        or lab_values[item][ind[1]] < medical_standard[item][0] * 0.8
                    )
                )
            ) and lab_variation:
                date = dates[ind[0]][:10] + " ~ " + dates[ind[1]][:10]
                if index_item.get(date):
                    index_item[date].append(
                        {
                            "item": item,
                            "date_index": [
                                ind[0] if ind[0] > 0 else 0,
                                (ind[1] if ind[1] < length - 1 else length - 1),
                            ],
                            "risk": risk_values[ind[1]],
                            "risk_variation": risk_values[ind[1]] / risk_values[ind[0]]
                            - 1,
                            "risk_variation_absolute": risk_values[ind[1]]
                            - risk_values[ind[0]],
                            "lab": lab_values[item][ind[1]],
                            "lab_variation": lab_variation,
                            "lab_variation_absolute": lab_values[item][ind[1]]
                            - lab_values[item][ind[0]],
                            "lab_attention": lab_attention_,
                            "lab_attention_variation": lab_attention_variation,
                        }
                    )
                else:
                    index_item[date] = [
                        {
                            "item": item,
                            "date_index": [
                                ind[0] if ind[0] > 0 else 0,
                                (ind[1] if ind[1] < length - 1 else length - 1),
                            ],
                            "risk": risk_values[ind[1]],
                            "risk_variation": risk_values[ind[1]] / risk_values[ind[0]]
                            - 1,
                            "risk_variation_absolute": risk_values[ind[1]]
                            - risk_values[ind[0]],
                            "lab": lab_values[item][ind[1]],
                            "lab_variation": lab_variation,
                            "lab_variation_absolute": lab_values[item][ind[1]]
                            - lab_values[item][ind[0]],
                            "lab_attention": lab_attention_,
                            "lab_attention_variation": lab_attention_variation,
                        }
                    ]

        for date, items in index_item.items():
            items = sorted(items, key=lambda x: x["lab_attention"], reverse=True)[:3]
            result.append(
                {
                    "type": 1,
                    "date": date,
                    "date_index": items[0]["date_index"],
                    "item": [item["item"] for item in items],
                    "risk": items[0]["risk"],
                    "risk_variation_absolute": items[0]["risk_variation_absolute"],
                    "risk_variation": items[0]["risk_variation"],
                    "lab": [item["lab"] for item in items],
                    "lab_variation": [item["lab_variation"] for item in items],
                    "lab_variation_absolute": [
                        item["lab_variation_absolute"] for item in items
                    ],
                    "lab_attention": [
                        item["lab_attention"] for item in items
                    ],
                    "lab_attention_variation": [
                        item["lab_attention_variation"] for item in items
                    ],
                }
            )

    items = []
    avgs = []
    attns = []
    trends = []
    for item in lab_names:
        flag2 = True
        for value in lab_attention[item]:
            if value < 0.1:
                flag2 = False
        if flag2:
            avg = sum(lab_values[item]) / length
            tmp_standard = medical_standard.copy()
            tmp_standard.update(
                {"Weight": [50, 80], "Appetite": [2000, 6000]}
            )
            if avg < tmp_standard[item][0]:
                items.append(item)
                avgs.append(avg)
                attns.append(sum(lab_attention[item]) / length)
                trends.append(-1)
            elif avg > tmp_standard[item][1]:
                items.append(item)
                avgs.append(avg)
                attns.append(sum(lab_attention[item]) / length)
                trends.append(1)

    if len(items) > 0:
        result.append(
            {
                "date": dates[0][:10] + " ~ " + dates[length - 1][:10],
                "date_index": [0, length - 1],
                "type": 2,
                "item": items,
                "lab_avg": avgs,
                "lab_attention": attns,
                "trend": trends,
            }
        )

    left = -1
    continued_high_risk_index = []
    for i, risk in enumerate(risk_values):
        if risk >= 0.5:
            if left < 0:
                left = i
        else:
            if left != -1 and i - left > 5:
                continued_high_risk_index.append([left, i - 1])
            left = -1
    if left != -1 and length - left >= 5:
        continued_high_risk_index.append([left, length - 1])
    if len(continued_high_risk_index) > 0:
        for index in continued_high_risk_index:
            items = []
            attns = []
            for item in lab_names:
                avg = sum(lab_attention[item][index[0] : index[1] + 1]) / (
                    index[1] - index[0] + 1
                )
                if avg >= 0.2:
                    items.append(item)
                    attns.append(avg)
            if len(items) > 0:
                result.append(
                    {
                        "date": dates[index[0]][:10] + " ~ " + dates[index[1]][:10],
                        "date_index": [index[0], index[1]],
                        "type": 3,
                        "item": items,
                        "lab_attention": attns,
                    }
                )
    return sorted(result, key=lambda x: x["date_index"][1])


if __name__ == "__main__":
    data_url = "CKD/processed"
    generate_prompt(data_url, 616)
        