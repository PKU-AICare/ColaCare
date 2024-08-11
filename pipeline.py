import json
import os
from tqdm import tqdm

from hparams import mimic_config as config
from agent import LeaderAgent, DoctorAgent
from utils.retrieve_utils import RetrievalSystem
from utils.runner_utils import *
from collaboration import *

print(config)

leader_agent = LeaderAgent(llm_name=config["llm_name"])

print("load leader agent over")

retrieval_system = RetrievalSystem(retriever_name=config["retriever_name"], corpus_name=config["corpus_name"])

doctor_agents = [DoctorAgent(dataset_name=config["ehr_dataset_name"], ehr_model_name=config['ehr_model_names'][i], seed=config["seeds"][i], mode=config["mode"], retrieval_system=retrieval_system, llm_name=config["llm_name"]) for i in range(config["doctor_num"])]

print("load doctor agents over")

result_json = {}

_, test_pids, _, train_y, test_y, val_y = load_data(config)

doctors = load_preds(config)

save_dir = f"./cost/{config['ehr_dataset_name']}_{config['mode']}_{'_'.join(config['ehr_model_names'])}_{config['llm_name']}_{config['corpus_name']}"

for patient_index, patient_id in tqdm(enumerate(test_pids), total=len(test_pids), desc=f"Processing patients in {config['ehr_dataset_name']} dataset"):
    sub_save_dir = f"{save_dir}/pid{patient_id}"

    # if os.path.exists(f"{sub_save_dir}/leader_final_summary.json") and os.path.exists(f"{sub_save_dir}/leader_logits_json.json"):
    #     continue

    os.makedirs(sub_save_dir, exist_ok=True)
    print('PatientID:', patient_id)
    analysis = []

    try:
        for i, doctor_agent in enumerate(doctor_agents):
            response, basic_context = doctor_agent.analysis(0, patient_id, f"{sub_save_dir}/doctor{i}_userprompt.json")
            json.dump(response, open(f"{sub_save_dir}/doctor{i}_analysis.json", "w"))
            analysis.append(response)
        json.dump(basic_context, open(f"{sub_save_dir}/basic_context.json", "w"))

        leader_agent.set_basic_info(basic_context)
        summary_content = leader_agent.summary(analysis, is_initial=True)
        json.dump(summary_content, open(f"{sub_save_dir}/leader_initial_summary.json", "w"))

        collaboration = Collaboration(leader_agent, doctor_agents, summary_content["Answer"], summary_content["Report"], sub_save_dir, doctor_num=config["doctor_num"], max_round=config["max_round"])
        current_report, _ = collaboration.collaborate()
        json.dump({"FinalReport": current_report}, open(f"{sub_save_dir}/leader_final_summary.json", "w"))
        
        result_json[patient_id] = current_report
        
        logits = leader_agent.revise_logits(analysis)
        logits['Ground Truth'] = 0 # test_y[patient_index][0][0]
        
        json.dump(logits, open(f"{sub_save_dir}/leader_logits_json.json", "w"))
    except Exception as e:
        print(f"Error in patient {patient_id}: {e}")
        result_json[patient_id] = "Error"
        continue
    
json.dump(result_json, open(f"{save_dir}/result.json", "w"))

print(len(result_json), "patients have been processed.")