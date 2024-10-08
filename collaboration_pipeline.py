import json
import os
import traceback as tb
import time
from tqdm import tqdm

from utils.hparams import mimic_config as config
from utils.framework import LeaderAgent, DoctorAgent, Collaboration
from utils.retrieve_utils import RetrievalSystem
from utils.runner_utils import *

print(config)

leader_agent = LeaderAgent(llm_name=config["llm_name"])

print("load leader agent over")

retrieval_system = RetrievalSystem(retriever_name=config["retriever_name"], corpus_name=config["corpus_name"])

doctor_agents = [DoctorAgent(config, i, retrieval_system) for i in range(config["doctor_num"])]

print("load doctor agents over")

result_json = {}
all_prompt_tokens = 0
all_completion_tokens = 0
total_start = time.time()

test_pids, test_y = load_data(config)

save_dir = f"./response/{config['ehr_dataset_name']}_{config['ehr_task']}_{config['mode']}_{'_'.join(config['ehr_model_names'])}_{config['llm_name']}_{config['corpus_name']}"
os.makedirs(save_dir, exist_ok=True)

for patient_index, patient_id in tqdm(enumerate(test_pids), total=len(test_pids), desc=f"Processing patients in {config['ehr_dataset_name']} dataset {config['ehr_task']} task {config['mode']} mode"):
    start = time.time()
    sub_save_dir = f"{save_dir}/pid{patient_id}"

    os.makedirs(sub_save_dir, exist_ok=True)
    print('PatientID:', patient_id)
    analysis = []
    prompt_tokens = 0
    completion_tokens = 0

    try:
        for i, doctor_agent in enumerate(doctor_agents):
            response, basic_context, messages, prompt_token, completion_token = doctor_agent.analysis(patient_index, patient_id)
            json.dump(response, open(f"{sub_save_dir}/doctor{i + 1}_review.json", "w"))
            analysis.append(response)
            prompt_tokens += prompt_token
            completion_tokens += completion_token
            json.dump(messages, open(f"{sub_save_dir}/doctor{i + 1}_review_messages.json", "w"))
            with open(f"{sub_save_dir}/doctor{i + 1}_review_userprompt.txt", "w") as f:
                f.write(messages[1]["content"])

        leader_agent.set_basic_info(basic_context)
        summary_content, messages, prompt_token, completion_token = leader_agent.summary(analysis, is_initial=True)
        json.dump(summary_content, open(f"{sub_save_dir}/meta_summary.json", "w"))
        json.dump(messages, open(f"{sub_save_dir}/meta_messages.json", "w"))
        with open(f"{sub_save_dir}/meta_userprompt.txt", "w") as f:
            f.write(messages[1]["content"])
        prompt_tokens += prompt_token
        completion_tokens += completion_token

        collaboration = Collaboration(leader_agent, doctor_agents, summary_content["answer"], summary_content["report"], sub_save_dir, doctor_num=config["doctor_num"], max_round=config["max_round"])
        current_report, _, prompt_token, completion_token = collaboration.collaborate()
        json.dump({"final_report": current_report}, open(f"{sub_save_dir}/meta_final_summary.json", "w"))
        prompt_tokens += prompt_token
        completion_tokens += completion_token
        
        logits, prompt_token, completion_token = leader_agent.revise_logits(analysis)
        if config["ehr_task"] == "outcome":
            logits['ground_truth'] = test_y[patient_index][-1][0]
        else:
            logits['ground_truth'] = test_y[patient_index][-1][2]
        json.dump(logits, open(f"{sub_save_dir}/leader_logits_json.json", "w"))
        prompt_tokens += prompt_token
        completion_tokens += completion_token
        
        end = time.time()
        result_json[patient_id] = {
            "report": current_report,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "time": f"{end - start:.2f} s"
        }
        
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
    except Exception as e:
        print(f"Error in patient {patient_id}")
        tb.print_exc()
        result_json[patient_id] = "Error"
        continue

result_json["total_prompt_tokens"] = all_prompt_tokens
result_json["total_completion_tokens"] = all_completion_tokens
total_end = time.time()
result_json["total_time"] = f"{total_end - total_start:.2f} s"
json.dump(result_json, open(f"{save_dir}/result.json", "w"))

print(len(result_json) - 3, "patients have been processed.")