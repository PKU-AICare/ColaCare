import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

# ## Preprocess for tuning
def run():
    text = json.load(open(text_dir, 'r'))
    selected_pids = list(text.keys())
    print('all', len(selected_pids))

    if mode == 'test':
        test_pids = pd.read_pickle(f"{ehr_dir}/test_pid.pkl")
        test_y = [y[0][0] for y in pd.read_pickle(f"{ehr_dir}/test_y.pkl")]
        ehr_embeddings = [pd.read_pickle(f"{ehr_dir}/{model_name}_seed{seed}_embeddings.pkl") for model_name, seed in zip(models, seeds)]
    else:
        test_pids = pd.read_pickle(f"{ehr_dir}/val_pid.pkl")
        test_y = [y[0][0] for y in pd.read_pickle(f"{ehr_dir}/val_y.pkl")]
        ehr_embeddings = [pd.read_pickle(f"{ehr_dir}/{model_name}_seed{seed}_embeddings2.pkl") for model_name, seed in zip(models, seeds)]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModel.from_pretrained("lm/GatorTron").to(device)
    tokenizer = AutoTokenizer.from_pretrained("lm/GatorTron")

    patients_data = {
        str(pid): {
            'label': int(label),
            'ehr_embeddings': [e for e in embeddings],
            'text_embedding': np.zeros((1024), dtype=np.float32),
        } for pid, label, embeddings in zip(test_pids, test_y, zip(*ehr_embeddings))
    }

    for pid, content in text.items():
        input_tokens = tokenizer(content,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=512,
            padding=True).to(device)
        with torch.no_grad():
            outputs = model(**input_tokens)
        text_embedding = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy()
        patients_data[pid].update({'text_embedding': text_embedding})

    os.makedirs(fusion_save_dir, exist_ok=True)

    if mode == 'test':
        test_patients = np.array(list(patients_data.keys()))
        test_text = [patients_data[patient_id]['text_embedding'] for patient_id in test_patients]
        test_ehr = [patients_data[patient_id]['ehr_embeddings'] for patient_id in test_patients]
        test_y = [patients_data[patient_id]['label'] for patient_id in test_patients]
        pd.to_pickle(test_text, f"{fusion_save_dir}/test_text.pkl")
        pd.to_pickle(test_ehr, f"{fusion_save_dir}/test_ehr.pkl")
        pd.to_pickle(test_y, f"{fusion_save_dir}/test_y.pkl")
        pd.to_pickle(test_patients, f"{fusion_save_dir}/test_pids.pkl")
    else:
        patients = np.array(list(patients_data.keys()))
        patients_outcome = [patients_data[patient_id]['label'] for patient_id in patients]
        train_patients, val_patients = train_test_split(patients, test_size=15/100, random_state=SEED, stratify=patients_outcome)

        train_text = [patients_data[patient_id]['text_embedding'] for patient_id in train_patients]
        train_ehr = [patients_data[patient_id]['ehr_embeddings'] for patient_id in train_patients]
        train_y = [patients_data[patient_id]['label'] for patient_id in train_patients]
        val_text = [patients_data[patient_id]['text_embedding'] for patient_id in val_patients]
        val_ehr = [patients_data[patient_id]['ehr_embeddings'] for patient_id in val_patients]
        val_y = [patients_data[patient_id]['label'] for patient_id in val_patients]

        print('train', len(train_patients), 'val', len(val_patients))
        print('selected in train', len([pid for pid in train_patients if pid in selected_pids]), 'selected in val', len([pid for pid in val_patients if pid in selected_pids]))

        pd.to_pickle(train_text, f"{fusion_save_dir}/train_text.pkl")
        pd.to_pickle(train_ehr, f"{fusion_save_dir}/train_ehr.pkl")
        pd.to_pickle(train_y, f"{fusion_save_dir}/train_y.pkl")
        pd.to_pickle(train_patients, f"{fusion_save_dir}/train_pids.pkl")
        pd.to_pickle(val_text, f"{fusion_save_dir}/val_text.pkl")
        pd.to_pickle(val_ehr, f"{fusion_save_dir}/val_ehr.pkl")
        pd.to_pickle(val_y, f"{fusion_save_dir}/val_y.pkl")
        pd.to_pickle(val_patients, f"{fusion_save_dir}/val_pids.pkl")

#------------------------parameters------------------------
dataset = 'esrd'
mode = 'test'
model_names = ['AdaCare', 'ConCare', 'RETAIN']
models = model_names
seed = 0
ehr_dir = f"ehr_datasets/{dataset}/processed/fold_1"
SEED = 42
seeds = [0, 0, 0]
text_dir = f"output/{dataset}_{mode}_{'_'.join(models)}_deepseek-chat_MSD/result.json"
fusion_save_dir = f"ehr_datasets/{dataset}/processed/fold_1/fusion"
run()