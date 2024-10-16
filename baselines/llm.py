from typing import List, Dict
import json

import numpy as np

from utils.framework import Agent, extract_and_parse_json
from utils.healthcare_context_utils import ContextBuilder
from baselines.baselines import *

class LLMAgent(Agent):
    def __init__(
        self,
        config,
        role: str = "doctor",
        dataset_name: str = "cdsl",
        ehr_model_name: str = "MCGRU",
        mode: str = "test",
        llm_name: str = "deepseek-chat",
    ) -> None:
        super().__init__(role, llm_name)
        
        self.ehr_task = config['ehr_task']
        self.context_builder = ContextBuilder(dataset_name=dataset_name, model_name=ehr_model_name, mode=mode,task=self.ehr_task)
        self.error_pids = []

    def single_llm_analysis(self, patient_index: int, patient_id: int, save_file: str, type:str) -> Dict[str, str]:
        """
        Returns:
            Dict[str, str | float]: {
                "logits": a float number between 0 and 1.
                "reason": doctor agent output part.
            }
        """
        healthcare_context = self.context_builder.generate_context(patient_index, patient_id, True)
        
        self.patient_info = healthcare_context
        
        if type=="zeroshot":
            system_prompt = zeroshot["system"]
            user_prompt = zeroshot["user"].render(hcontext=healthcare_context)
        elif type=="fewshot":
            system_prompt = fewshot["system"]
            user_prompt = fewshot['user'].render(hcontext=healthcare_context)
        elif type=="sc":
            system_prompt = sc["system"]
            user_prompt = sc['user'].render(hcontext=healthcare_context)
        # elif type=="debate":
        #     system_prompt = zeroshot["system"]
        #     user_prompt = zeroshot['user'].render(hcontext=healthcare_context)
        # elif type=="specialist":
        #     system_prompt = specialist["system"]
        #     user_prompt = specialist['user'].render(hcontext=healthcare_context)
        # elif type=="confidence":
        #     system_prompt = confidence["system"]
        #     user_prompt = confidence['user'].render(hcontext=healthcare_context)
        else:
            raise ValueError("type must be one of 'zero shot', 'few shot', 'self consistency'")

        ans = self.invoke(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        if ans['reason'] == "No valid JSON content found.":
            self.error_pids.append(patient_id)
            
        if type=="zeroshot" or type=="fewshot":
            self.latest_analysis = ans["reason"]
            self.latest_logits = ans["logits"]
        elif type=="sc":
            self.latest_analysis = ans["reason"]
            self.latest_logits = ans["logits"]

        json.dump(user_prompt, open(save_file, "w"))
        
        return ans, healthcare_context

    def get_error_pids(self):
        return self.error_pids
    
    def meta_judge(self, doctor_responses):
        system_prompt = specialist_collaboration['system']
        user_prompt = specialist_collaboration['user'].render(reports=doctor_responses)
        
        ans = self.invoke(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        self.latest_logits = ans["revised_logits"]
        self.latest_analysis = ans["revised_report"]
        return ans
        
    def simple_collaboration(self, other_opinions: List[str],type: str) -> Dict[str, str]:
        if type=="debate":
            system_prompt = debate_collaboration['system']
            user_prompt = debate_collaboration['user'].render(logit=self.latest_logits, analysis=self.latest_analysis, reports=other_opinions)
        elif type=="specialist":
            system_prompt = debate_collaboration['system']
            user_prompt = debate_collaboration['user'].render(logit=self.latest_logits, analysis=self.latest_analysis, reports=other_opinions)
        elif type=="confidence":
            system_prompt = confidence_collaboration['system']
            user_prompt = confidence_collaboration['user'].render(logit=self.latest_logits, analysis=self.latest_analysis, reports=other_opinions) 
        else:
            raise ValueError("type must be one of 'debate', 'specialist', 'confidence'")
                   
        ans = self.invoke(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        self.latest_logits = ans["revised_logits"]
        self.latest_analysis = ans["revised_report"]
        return ans

    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, str]:

        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=False
        )
        content = response.choices[0].message.content
        ans = extract_and_parse_json(content)
        return ans


class SimpleCollaboration():
    def __init__(self, doctor_agents: List[LLMAgent], meta_agent: LLMAgent, analysis: List[str], save_dir: str, doctor_num: int=2, max_round: int=2,type="debate"):
        self.doctor_responses = analysis
        self.doctor_num = doctor_num
        self.doctor_agents = doctor_agents
        self.meta_agent = meta_agent
        self.max_round = max_round
        self.now_round = 0
        self.save_dir = save_dir
        self.type=type

    def get_doctor_opinions(self,doctor_responses):
        doctor_opinions = []
        for i in range(len(doctor_responses)):
            logits = "This doctor thinks the death rate of the patient is "+ str(doctor_responses[i]['revised_logits'])+". "
            reasons = "The reason to support his judgement is: "+doctor_responses[i]['revised_report']
            doctor_opinions.append(logits+reasons)
        return doctor_opinions
    
    def collaborate(self):
        # assert len(self.doctor_responses) == 0 and self.now_round == 0
        result={}
        while self.now_round < self.max_round:
            new_responses = []
            new_results = []
            self.now_round += 1
            for i, doctor in enumerate(self.doctor_agents):
                analysis = []
                for j in range(len(self.doctor_responses)):
                    if i == j:
                        continue
                    analysis.append(self.doctor_responses[j])
                
                response = doctor.simple_collaboration(analysis,type=self.type)
                new_responses.append(response["revised_report"])
                new_results.append(response)
                json.dump(response, open(f"{self.save_dir}/doctor{i}_round{self.now_round}_analysis.json", "w"))
            self.doctor_responses = new_responses
            self.doctor_results = new_results
            
        # if self.now_round == self.max_round:
        #     print("No agreement has been reached")
        
        self.now_round = 0
        
        # print(len(self.doctor_responses))
        # print(self.doctor_responses[0])
        
        if self.type == "debate":
            result['logits']=np.mean([float(response['revised_logits']) for response in self.doctor_results])
            result['reasons']=[response['revised_report'] for response in self.doctor_results]
        elif self.type == "confidence":
            confidence_weights = [float(response['confidence']) for response in self.doctor_results]
            confidence_weights = np.array(confidence_weights)/np.sum(confidence_weights)
            logits = [float(response['revised_logits']) for response in self.doctor_results]
            result['logits'] = np.sum(np.dot(confidence_weights, logits))
            result['reasons']=[response['revised_report'] for response in self.doctor_results]
            result['confidence'] = [response['confidence'] for response in self.doctor_results]
        elif self.type == "specialist":
            doctor_opinions = self.get_doctor_opinions(self.doctor_results)
            result_meta=self.meta_agent.meta_judge(doctor_opinions)
            result['logits'] = result_meta['revised_logits']
            result['reasons'] = result_meta['revised_report']
        else:
            raise ValueError("Invalid type")
        
        return result, self.now_round