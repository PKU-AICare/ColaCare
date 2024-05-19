import json
import os

import pandas as pd

from models.ckd_utils import generate_prompt
from generator import Generator


data_url = "datasets/CKD"

# cases = json.load(open(f'{data_url}/cases.json', 'r'))
pids = pd.read_pickle(f'{data_url}/pid.pkl')
for patient_id in pids:
    generator = Generator()
    print(f'Patient ID: {patient_id}')
    try:
        hcontext = generate_prompt(data_url, patient_id)
    except Exception as e:
        print(f"Patient {patient_id} failed with error: {e}")
        continue

    save_dir=f'./response/pid{patient_id}'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/hcontext.txt', 'w') as f:
        f.write(hcontext)
    try:
        answer, keywords, snippets, scores, messages = generator.generate_answer(hcontext=hcontext)
    except Exception as e:
        print(f"Patient {patient_id} failed with error: {e}")
        continue
    with open(f'{save_dir}/keywords.txt', 'w') as f:
        f.write(keywords)
    with open(f'{save_dir}/snippets.json', 'w') as f:
        json.dump(snippets, f)
    with open(f'{save_dir}/scores.json', 'w') as f:
        json.dump(scores, f)
    with open(f'{save_dir}/messages.json', 'w') as f:
        json.dump(messages, f)
    with open(f'{save_dir}/response.json', 'w') as f:
        json.dump(answer, f)