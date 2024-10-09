import json
import os
from tqdm import tqdm

from hparams import mimic_config as config
from agent import LeaderAgent, DoctorAgent, LLMAgent
from utils.retrieve_utils import RetrievalSystem
from utils.runner_utils import *
from collaboration import *

single_type="fewshot"
collaboration_type="debate"

print(config)

doctor_agents = [LLMAgent(dataset_name=config["ehr_dataset_name"], ehr_model_name=config['ehr_model_names'][i], seed=config["seeds"][i], mode=config["mode"], llm_name=config["llm_name"]) for i in range(config["doctor_num"])]

print("load doctor agents over")

result_json = {}

_, test_pids, _, train_y, test_y, val_y = load_data(config)

save_dir = f"./baseline_results/{config['ehr_dataset_name']}_{config['mode']}_{config['llm_name']}_{config['corpus_name']}/{single_type}_{collaboration_type}"
os.makedirs(save_dir, exist_ok=True)

for patient_index, patient_id in tqdm(enumerate(test_pids), total=len(test_pids), desc=f"Processing patients in {config['ehr_dataset_name']} dataset"):
    sub_save_dir = f"{save_dir}/pid{patient_id}"

    if patient_index > 3:
        break

    os.makedirs(sub_save_dir, exist_ok=True)
    print('PatientID:', patient_id)
    analysis = []
    analysis_result = {"reasons":[],'logits':[]}
    # try:
    for i, doctor_agent in enumerate(doctor_agents):
        response, basic_context = doctor_agent.single_llm_analysis(patient_index, patient_id, f"{sub_save_dir}/doctor{i}_userprompt.json",type=single_type)
        json.dump(response, open(f"{sub_save_dir}/doctor{i}_analysis.json", "w"))
        analysis.append(response["reason"])
        analysis_result["reasons"].append(response["reason"])
        analysis_result["logits"].append(response["logits"])

    json.dump(analysis_result, open(f"{sub_save_dir}/doctor_analysis.json", "w"))
    json.dump(basic_context, open(f"{sub_save_dir}/basic_context.json", "w"))

    collaboration = SimpleCollaboration(doctor_agents, analysis, sub_save_dir, doctor_num=2, max_round=2,type=collaboration_type)
    current_report, _ = collaboration.collaborate()
    json.dump({"final_report": current_report}, open(f"{sub_save_dir}/final_summary.json", "w"))
    
    result_json[patient_id] = current_report
    
    logits = {}
    logits['predicted_result'] = float(current_report['logits'])
    logits['ground_truth'] = test_y[patient_index][0][0]
    json.dump(logits, open(f"{sub_save_dir}/logits_json.json", "w"))
    
    # except Exception as e:
    #     print(f"Error in patient {patient_id}: {e}")
    #     result_json[patient_id] = "Error"
    #     continue
    
json.dump(result_json, open(f"{save_dir}/result.json", "w"))

print(len(result_json), "patients have been processed.")