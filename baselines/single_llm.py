import json
import os
from tqdm import tqdm

from utils.hparams import mimic_config as config
from agent import LeaderAgent, DoctorAgent, LLMAgent
from utils.retrieve_utils import RetrievalSystem
from utils.runner_utils import *
from collaboration import *

baseline_type="sc"

print(config)

doctor_agent = LLMAgent(config,dataset_name=config["ehr_dataset_name"], ehr_model_name=config['ehr_model_names'][0], mode=config["mode"], llm_name=config["llm_name"])

result_json = {}
                    
test_pids, test_y = load_data(config)

save_dir = f"./baseline_results_old/{config['ehr_dataset_name']}_{config['mode']}_{config['llm_name']}_{config['corpus_name']}/{baseline_type}"
os.makedirs(save_dir, exist_ok=True)
# import random
# random.seed(42)
# # random choice 100 different from patients
# picked_pids = set(random.sample(test_pids, 100))
picked_pids = {'19109196', '12266059', '16277409', '15366618', '10878442', '15921538', '14971764', '17554265', '17021161', '10856048', '11330416', '18822469', '19421824', '18474441', '18430220', '10451157', '19314496', '19536226', '17436451', '15267698', '16395973', '18779408', '14311242', '18713323', '10943603', '18741255', '16584718', '11395953', '13127430', '16619623', '11957937', '16707579', '13629419', '15906818', '15642676', '16047293', '16607138', '10255006', '19265807', '15866889', '18122533', '17538197', '12797755', '17841401', '12870687', '16360107', '13519869', '13396394', '17956685', '15501368', '13316366', '10101116', '18769537', '15320468', '15274195', '18558073', '11068569', '12414025', '13223351', '18026284', '10438951', '16353835', '13253950', '18585326', '13877262', '10254291', '14730920', '16598054', '12746511', '12904593', '10874416', '14371185', '10652321', '14472165', '10433353', '15801921', '13233757', '16809653', '10331279', '18976221', '18794248', '17179127', '10801437', '15533649', '12729806', '16154215', '18552025', '17375855', '18959691', '15886609', '11773170', '12612603', '19482688', '13227181', '10844468', '12739166', '17525053', '16292532', '19774387', '16947984'}

# print(picked_pids)
# print(len(picked_pids))
# exit(0)
for patient_index, patient_id in tqdm(enumerate(test_pids), total=len(test_pids), desc=f"Processing patients in {config['ehr_dataset_name']} dataset"):
    
    if patient_id in picked_pids:
        # print(patient_id)
        continue
    
    if patient_index <= 100:
        continue
    
    sub_save_dir = f"{save_dir}/pid{patient_id}"
    print(sub_save_dir)
    
    os.makedirs(sub_save_dir, exist_ok=True)
    print('PatientID:', patient_id)
    analysis = []

    # try:
    response, health_context = doctor_agent.single_llm_analysis(patient_index, patient_id, f"{sub_save_dir}/doctor_userprompt.json",type=baseline_type)

    print(sub_save_dir)
    # print(f"{sub_save_dir}/doctor_analysis.json")
    
    json.dump(response, open(f"{sub_save_dir}/doctor_analysis.json", "w"))
    json.dump(health_context, open(f"{sub_save_dir}/basic_context.json", "w"))

    result_json[patient_id] = response

    logits = {}
    logits['predicted_result'] = float(response['logits'])
    logits['ground_truth'] = test_y[patient_index][0][2]
    
    json.dump(logits, open(f"{sub_save_dir}/logits_json.json", "w"))
    
    # except Exception as e:
    #     print(f"Error in patient {patient_id}: {e}")
    #     result_json[patient_id] = "Error"
    #     continue
        
json.dump(result_json, open(f"{save_dir}/result.json", "w"))

print(doctor_agent.get_error_pids)

print(len(result_json), "patients have been processed.")