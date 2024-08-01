import json
import os
from tqdm import tqdm

# from hparams import config
from hparams import mimic_config as config 
from agent import LeaderAgent, DoctorAgent
from utils.retrieve_utils import RetrievalSystem
from utils.runner_utils import *
from collaboration import *

leader_agent = LeaderAgent(llm_name=config["llm_name"])

print("load leader agent over")

retrieval_system = RetrievalSystem(retriever_name=config["retriever_name"], corpus_name=config["corpus_name"])

doctor_agents = [DoctorAgent(dataset_name=config["ehr_dataset_name"], ehr_model_name=config['ehr_model_names'][i], seed=config["seeds"][i], retrieval_system=retrieval_system, llm_name=config["llm_name"]) for i in range(config["doctor_num"])]

print("load doctor agents over")

result_json = {}

# test_index = [27, 36, 56, 73, 82, 125, 135, 151, 193, 205, 243, 248, 252, 264, 268, 274, 283, 398, 406]
# test_pids = [299, 419, 710, 848, 903, 1373, 1498, 1673, 2077, 2208, 2560, 2648, 2694, 2780, 2819, 2842, 2942, 4154, 4228]

_, test_pids, _, train_y, test_y, val_y = load_data(config['ehr_dataset_dir'])

doctors = load_preds(config)

for patient_index, patient_id in tqdm(enumerate(test_pids[:100]), total=len(test_pids[:100]), desc=f"Processing patients in {config['ehr_dataset_name']} dataset"):
# for patient_index, patient_id in tqdm(zip(test_index[:1], test_pids[:1]), total=len(test_pids[:1]), desc=f"Processing patients in {config['ehr_dataset_name']} dataset"):
    save_dir = f"./output/{config['ehr_dataset_name']}_{'_'.join(config['ehr_model_names'])}_{config['llm_name']}_{config['corpus_name']}/pid{patient_id}"
    os.makedirs(save_dir, exist_ok=True)

    if not check_numbers([doctor[patient_index] for doctor in doctors]):
        logits = {}
        for i in range(len(doctors)):
            logits["Doctor " + str(i) + "'s Logit"] = doctors[i][patient_index]
        logits.update({
            "Final Logit": sum([doctors[i][patient_index] for i in range(len(doctors))]) / len(doctors),
            "Ground Truth": test_y[patient_index][0][0]
        })
        json.dump(logits, open(f"{save_dir}/leader_logits_json.json", "w"))
        continue
        
    analysis = []
    try:
        for i, doctor_agent in enumerate(doctor_agents):
            response, basic_context = doctor_agent.analysis(patient_index, patient_id)
            json.dump(response, open(f"{save_dir}/doctor{i}_analysis.json", "w"))
            analysis.append(response)

        leader_agent.set_basic_info(basic_context)
        summary_content = leader_agent.summary(analysis, is_initial=True)
        json.dump(summary_content, open(f"{save_dir}/leader_initial_summary.json", "w"))

        collaboration = Collaboration(leader_agent, doctor_agents, summary_content["Answer"], summary_content["Report"], save_dir, doctor_num=config["doctor_num"], max_round=config["max_round"])
        current_report, _ = collaboration.collaborate()
        json.dump({"FinalReport": current_report}, open(f"{save_dir}/leader_final_summary.json", "w"))
        
        result_json[patient_id] = current_report
        
        logits = leader_agent.revise_logits(analysis)
        logits['Ground Truth'] = test_y[patient_index][0][0]
        
        json.dump(logits, open(f"{save_dir}/leader_logits_json.json", "w"))
    except Exception as e:
        print(f"Error in patient {patient_id}: {e}")
        result_json[patient_id] = "Error"
        continue
    
json.dump(result_json, open(f"./output/{config['ehr_dataset_name']}_{'_'.join(config['ehr_model_names'])}_{config['llm_name']}_{config['corpus_name']}/result.json", "w"))