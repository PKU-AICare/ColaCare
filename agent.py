from typing import Literal, List, Dict
import re
import json

from openai import OpenAI

from agent import *
from utils.retrieve_utils import RetrievalSystem
from utils.healthcare_context_utils import ContextBuilder
from utils.config import deep_config, tech_config
from utils.template import *


class Agent:
    def __init__(
        self,
        role: Literal["doctor", "leader"],
        ehr_model_name: str = "MCGRU",
        llm_name: str = "deepseek-chat"
    ) -> None:
        self.role = role
        self.ehr_model_name = ehr_model_name
        self.llm_name = llm_name
        if llm_name == "deepseek-chat":
            self.llm_config = deep_config
        else:
            self.llm_config = tech_config
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        
    def clear_history(self):
        self.conversation_history.clear()


class DoctorAgent(Agent):
    def __init__(
        self,
        role: str = "doctor",
        dataset_name: str = "cdsl",
        ehr_model_name: str = "MCGRU",
        seed: int = 1,
        retrieval_system: RetrievalSystem = None,
        llm_name: str = "deepseek-chat",
    ) -> None:
        super().__init__(role, ehr_model_name, llm_name)
        
        self.client = OpenAI(api_key=self.llm_config["api_key"], base_url=self.llm_config["api_base"])
        self.retrieval_system = retrieval_system
        self.context_builder = ContextBuilder(dataset_name=dataset_name, model_name=ehr_model_name, seed=seed)

    def analysis(self, patient_index: int, patient_id: int) -> Dict[str, str]:
        basic_context, subcontext, healthcare_context = self.context_builder.generate_context(patient_index, patient_id)
        context = self.retrieve(subcontext, k=16)
        
        self.patient_info = healthcare_context
        
        system_prompt = doctor_analysis_system
        user_prompt = doctor_analysis_user.render(context=context, hcontext=healthcare_context)
        ans = self.invoke(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        self.latest_analysis = ans["Analysis"]
        self.initial_analysis = ans["Analysis"]
        return ans, basic_context
    
    def collaboration(self, leader_opinion, leader_report) -> Dict[str, str]:
        """Doctor agent collaboration with leader agent.

        Returns:
            Dict[str, str | float]: {
                "Answer": agree / disagree.
                "Confidence": a float number between 0 and 1.
                "Reason": doctor agent output part.
                "Evidences": doctors' new rag documents.
            }
        """
        context = self.retrieve(self.latest_analysis, k=16)

        system_prompt = doctor_collaboration_system
        user_prompt = doctor_collaboration_user.render(context=context, analysis=self.initial_analysis, opinion=leader_opinion, report=leader_report)
        
        ans = self.invoke(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        self.latest_analysis = ans["Reason"] + ("\n".join(ans["Evidences"]) if "Evidences" in ans else "")
        return ans

    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        # for message in messages:
        #     self.add_message(message["role"], message["content"])

        response = self.client.chat.completions.create(
            model=self.llm_name,
            # messages=self.conversation_history,
            messages=messages,
            stream=False
        )
        content = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        print(prompt_tokens, completion_tokens)
        # self.add_message("assistant", content)
        ans = extract_and_parse_json(content)
        return ans

    def retrieve(self, question: str, k: int=16) -> str:
        retrieve_texts, _, _ = self.retrieval_system.retrieve(question, k=k)
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieve_texts[idx]["title"], retrieve_texts[idx]["content"]) for idx in range(len(retrieve_texts))]
        context = "\n".join(contexts)
        return context


class LeaderAgent(Agent):
    def __init__(
        self,
        role: str = "leader",
        ehr_model_name: str = "MCGRU",
        llm_name: str = "deepseek-chat",
    ) -> None:
        super().__init__(role, ehr_model_name, llm_name)
        self.action_system_prompt = action_system
        self.action_user_prompt = action_user
        self.client = OpenAI(api_key=self.llm_config["api_key"], base_url=self.llm_config["api_base"])
        self.current_report = ""
        self.basic_info = None

    def set_basic_info(self, patient_info: str) -> None:
        self.basic_info = patient_info

    def generate_doctors_prompt(self, doctor_responses: List[Dict], is_initial=False) -> str:
        responses = ""
        for idx in range(len(doctor_responses)):
            i = doctor_responses[idx]
            if is_initial:
                response=initial_doctor_summary.render(diagnosis=i['Logit'], analysis=i['Analysis'], evidence='\n'.join(i['Evidences']))
            else:
                if i['Answer'].lower()=="agree" and i['Confidence']>=0.5:
                    response=doctor_vote_agree_certain.render(reason=i['Reason'])
                elif i['Answer'].lower()=="agree" and i['Confidence']<0.5:
                    response=doctor_vote_agree_uncertain.render(reason=i['Reason'])
                elif i['Answer'].lower()=="disagree" and i['Confidence']>=0.5:
                    response=doctor_vote_disagree_certain.render(reason=i['Reason'], documents='\n'.join(i['Evidences']))
                elif i['Answer'].lower()=="disagree" and i['Confidence']<0.5:
                    response=doctor_vote_disagree_uncertain.render(reason=i['Reason'], documents='\n'.join(i['Evidences']))
                else:
                    print("Doctor Vote Templete WRONG!!!")
            if idx < len(doctor_responses)-1:
                response = "Doctor "+str(idx)+"'s statement is as following:\n"+response+"\n"
            else:
                response = "Doctor "+str(idx)+"'s statement is as following:\n"+response
            responses = responses + response
        return responses

    def extract_action_response(self, response: Dict):
        if "not continue" in response['Action'].lower() or "stop" in response['Action'].lower():
            return 0
        else:
            return 1

    def next_action(self, doctor_responses):
        """
        doctor_responses的输入形式:[
            {
                "Answer": agree / disagree
                "Confidence": 0 - 1
                "Reason": doctor llm output part
                "Evidences": doctors' new rag documents
            },
            ...
        ]
        输出: 1 代表继续循环; 0 代表结束循环
        """
        action_user_prompt=self.generate_doctors_prompt(doctor_responses)
        
        messages=[
            {"role": "system", "content": action_system},
            {"role": "user", "content": action_user.render(opinions=action_user_prompt)}
        ]
    
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=False
        )
        content = response.choices[0].message.content
        content = extract_and_parse_json(content)
        
        ans = self.extract_action_response(content)
        
        return ans, content
    
    def summary(self, doctor_responses, is_initial=False) -> str:
        """ 
        依据：
        1. patient_info: 病人基本信息
        2. doctor_answer: 整理后的doctor_responses
        3. current_report: 上一轮的病历总结
        (agreed_evidence: 被认可的证据列表)
        """
        doctors_answer_prompt = self.generate_doctors_prompt(doctor_responses, is_initial)
        
        if is_initial:
            messages=[
                {"role": "system", "content": initial_summary_system},
                {"role": "user", "content": initial_summary_user.render(patient_info=self.basic_info, doctor_info=doctors_answer_prompt)}
            ]
        else:
            messages=[
                {"role": "system", "content": collaborative_summary_system},
                {"role": "user", "content": collaborative_summary_user.render(doctor_info=doctors_answer_prompt, latest_info=self.current_report)}
            ]
        
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=False
        )

        content = response.choices[0].message.content
        content = extract_and_parse_json(content)

        self.current_report = content['Answer']+"\n"+content['Report'] 
        
        return content
    
    def generate_doctors_initial_prompt(self, doctor_responses: List[Dict]) -> str:
        responses = ""
        for idx in range(len(doctor_responses)):
            i = doctor_responses[idx]
            prompt = initial_doctor_for_revise.render(diagnosis=i['Logit'], analysis=i['Analysis'])
            if idx < len(doctor_responses)-1:
                response = "Doctor "+str(idx)+"'s statement is as following:\n"+prompt+"\n"
            else:
                response = "Doctor "+str(idx)+"'s statement is as following:\n"+prompt
            responses = responses + response
        return responses
    
    def revise_logits(self,doctor_initial_responses):
        
        doctors_answer_prompt = self.generate_doctors_initial_prompt(doctor_initial_responses)
        
        messages = [
            {"role": "system", "content": revise_logits_system},
            {"role": "user", "content": revise_logits_user.render(doctor_info=doctors_answer_prompt, latest_info=self.current_report)}
        ]
        
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=False
        )

        content = response.choices[0].message.content
        content = extract_and_parse_json(content)

        # logits = content['Final Logit']

        return content


def extract_and_parse_json(text: str):
    pattern_backticks = r'```json(.*?)```'
    match = re.search(pattern_backticks, text, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(json_string)
            raise ValueError("Invalid JSON content found.")
    
    match = re.search(pattern_backticks, text.strip() + "\"]}```", re.DOTALL)
    if match:
        json_string = match.group(1).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(json_string)
            raise ValueError("Invalid JSON content found.")
    
    pattern_json_object = r'\{.*?\}'
    match = re.search(pattern_json_object, text, re.DOTALL)
    if match:
        json_string = match.group(0).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(json_string)
            raise ValueError("Invalid JSON content found.")
    
    print(text)
    raise ValueError("No valid JSON content found.")