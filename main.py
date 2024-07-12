import json
import os
from tqdm import tqdm

import pandas as pd

from models.ckd_utils import generate_prompt
from retcare import RetCare
from keywords_utils import generate_keywords, extract_and_parse_json

dataset = "cdsl"
data_url = f"ehr_datasets/{dataset}/processed/fold_1"
retriever_name="MedCPT"
corpus_name="Textbooks"
llm_model = "deepseek-chat"
retcare = RetCare(llm_name=llm_model, ensemble='select', retriever_name=retriever_name, corpus_name=corpus_name)
pids = pd.read_pickle(f'{data_url}/test_pid.pkl')

for patient_index, patient_id in tqdm(enumerate(pids[:2]), total=len(pids[:2])):
    try:
        subcontext, hcontext = generate_prompt(dataset, data_url, patient_index, patient_id)
    except Exception as e:
        print(f"Patient {patient_id} failed with error: {e}")
        continue

    try:
        keywords = generate_keywords(llm_model, subcontext)
        keywords = ', '.join(keywords['keywords'])
    except Exception as e:
        print(f"Patient {patient_id} failed in generating keywords with error: {e}")
        continue

    save_dir=f'./response/{dataset}/pid{patient_id}'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/hcontext.txt', 'w') as f:
        f.write(hcontext)

    try:
        answer, snippets, scores, messages = retcare.answer(hcontext=hcontext, keywords=keywords, k=20)
    except Exception as e:
        print(f"Patient {patient_id} failed in answering with error: {e}")
        continue
    
    with open(f'{save_dir}/snippets.json', 'w') as f:
        json.dump(snippets, f)
    with open(f'{save_dir}/scores.json', 'w') as f:
        json.dump(scores, f)
    with open(f'{save_dir}/messages.json', 'w') as f:
        json.dump(messages, f)
    with open(f'{save_dir}/response.txt', 'w') as f:
        f.write(answer)
    with open(f'{save_dir}/response.json', 'w') as f:
        try:
            answer = extract_and_parse_json(answer)
        except Exception as e:
            answer = {"answer": answer, "error": str(e)}
        json.dump(answer, f)