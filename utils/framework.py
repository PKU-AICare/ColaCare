from typing import Literal, List, Dict
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from utils.framework import *
from utils.retrieve_utils import RetrievalSystem
from utils.healthcare_context_utils import ContextBuilder
from utils.config import deep_config, v1_config, v2_config, default_config
from utils.prompt_template.template import *


class Agent:
    def __init__(
        self,
        role: Literal["doctor", "leader"],
        llm_name: str = "deepseek-chat"
    ) -> None:
        self.role = role
        self.llm_name = llm_name
        if llm_name == "deepseek-chat":
            self.llm_config = deep_config
        elif "gpt" in llm_name.lower():
            self.llm_config = v1_config
        elif "qwen" in llm_name.lower() or "doubao" in llm_name.lower() or "claude" in llm_name.lower():
            self.llm_config = v2_config
        elif "llama" in llm_name.lower():
            self.llm_config = default_config
        self.client = OpenAI(
            api_key=self.llm_config["api_key"], base_url=self.llm_config["api_base"])
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def clear_history(self):
        self.conversation_history.clear()


class DoctorAgent(Agent):
    def __init__(
        self,
        config: Dict,
        index: int = 0,
        retrieval_system: RetrievalSystem = None,
        role: str = "doctor",
    ) -> None:
        super().__init__(role, config["llm_name"])

        self.dataset_name = config['ehr_dataset_name']
        self.ehr_task = config['ehr_task']
        self.ehr_model_name = config['ehr_model_names'][index]
        self.mode = config['mode']
        self.retrieval_system = retrieval_system
        self.context_builder = ContextBuilder(dataset_name=self.dataset_name, task=self.ehr_task, model_name=self.ehr_model_name, mode=self.mode)

    def analysis(self, patient_index: int, patient_id: int) -> tuple[Dict[str, str], str, int, int]:
        basic_context, subcontext, healthcare_context = self.context_builder.generate_context(patient_index, patient_id)
        context = self.retrieve(subcontext)
        self.patient_info = healthcare_context

        system_prompt = doctor_review_system
        user_prompt = doctor_review_user.render(context=context, hcontext=healthcare_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        ans, prompt_token, completion_token = self.invoke(messages=messages)
        self.latest_analysis = ans["analysis"]
        self.initial_analysis = ans["analysis"]

        return ans, basic_context, messages, prompt_token, completion_token

    def collaboration(self, leader_opinion, leader_report) -> tuple[Dict[str, str], int, int]:
        """Doctor agent collaboration with leader agent.

        Returns:
            Dict[str, str | float]: {
                "answer": agree / disagree.
                "confidence": a float number between 0 and 1.
                "reason": doctor agent output part.
                "evidences": doctors' new rag documents.
            }
        """
        context = self.retrieve(self.latest_analysis)

        system_prompt = doctor_collaboration_system
        user_prompt = doctor_collaboration_user.render(context=context, analysis=self.latest_analysis, opinion=leader_opinion, report=leader_report)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        ans, prompt_token, completion_token = self.invoke(messages=messages)
        self.latest_analysis = ans["reason"] + ("\n".join(ans["evidences"]) if "evidences" in ans else "")
        return ans, messages, prompt_token, completion_token

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                stream=False
            )
            content = response.choices[0].message.content
            prompt_token = response.usage.prompt_tokens
            completion_token = response.usage.completion_tokens
            ans = extract_and_parse_json(content)
        except Exception as e:
            raise e
        end = time.time()
        print(f"DoctorAgent time cost: {end - start:.2f}s.")
        return ans, prompt_token, completion_token

    def retrieve(self, question: str, k: int = 3) -> str:
        retrieve_texts, _, _ = self.retrieval_system.retrieve(question, k=k)
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx + 1, retrieve_texts[idx]["title"], retrieve_texts[idx]["content"]) for idx in range(len(retrieve_texts))]
        context = "\n".join(contexts)
        return context


class LeaderAgent(Agent):
    def __init__(
        self,
        role: str = "leader",
        llm_name: str = "deepseek-chat",
    ) -> None:
        super().__init__(role, llm_name)
        self.action_system_prompt = action_system
        self.action_user_prompt = action_user
        self.current_report = ""
        self.basic_info = None

    def set_basic_info(self, patient_info: str) -> None:
        self.basic_info = patient_info

    def generate_doctors_prompt(self, doctor_responses: List[Dict], is_initial=False) -> str:
        responses = ""
        for idx, res in enumerate(doctor_responses):
            if is_initial:
                if 'evidences' in res:
                    evidence = res['evidences']
                    if not isinstance(evidence[0], str):
                        evidence = list(map(str, evidence))
                    evidence = '\n'.join(evidence)
                else:
                    evidence = ""
                response = initial_doctor_summary.render(diagnosis=res['logit'], analysis=res['analysis'], evidence=evidence)
            else:
                documents = '\n'.join(res['evidences']) if 'evidences' in res else ""
                if 'confidence' not in res or int(res['confidence']) == 1:
                    response = doctor_low_confidence.render(reason=res['reason'], opinion=res['answer'], documents=documents)
                elif int(res['confidence']) == 2:
                    response = doctor_moderate_confidence.render(reason=res['reason'], opinion=res['answer'], documents=documents)
                elif int(res['confidence']) == 3:
                    response = doctor_high_confidence.render(reason=res['reason'], opinion=res['answer'], documents=documents)
                else:
                    print(f"Doctor Confidence Error: {res['confidence']}!")
            response = "Doctor " + str(idx + 1) + "'s statement is as following:\n" + response + "\n"
            responses = responses + response
        return responses

    def extract_action_response(self, response: Dict):
        if "yes" in response['action'].lower():
            return 1
        elif "no" in response['action'].lower():
            return 0
        else:
            raise ValueError(f"Invalid action response: {response['action']}.")

    def next_action(self, doctor_responses) -> tuple[int, Dict[str, str], int, int]:
        """
        doctor_responses的输入形式:[
            {
                "answer": agree / disagree
                "confidence": 0 - 1
                "reason": doctor llm output part
                "evidences": doctors' new rag documents
            },
            ...
        ]
        输出: 1 代表继续循环; 0 代表结束循环
        """
        action_user_prompt = self.generate_doctors_prompt(doctor_responses)
        messages = [
            {"role": "system", "content": action_system},
            {"role": "user", "content": action_user.render(opinions=action_user_prompt)}
        ]
        ans, prompt_token, completion_token = self.invoke(messages)
        action = self.extract_action_response(ans)
        return action, ans, messages, prompt_token, completion_token

    def summary(self, doctor_responses, is_initial=False) -> tuple[Dict[str, str], str, int, int]:
        """ 
        依据：
        1. patient_info: 病人基本信息
        2. doctor_answer: 整理后的doctor_responses
        3. current_report: 上一轮的病历总结
        (agreed_evidence: 被认可的证据列表)
        """
        doctors_answer_prompt = self.generate_doctors_prompt(doctor_responses, is_initial)
        if is_initial:
            messages = [
                {"role": "system", "content": meta_summary_system},
                {"role": "user", "content": meta_summary_user.render(patient_info=self.basic_info, doctor_info=doctors_answer_prompt)}
            ]
            ans, prompt_token, completion_token = self.invoke(messages)
            self.current_report = ans['answer'] + "\n" + ans['report']
            return ans, messages, prompt_token, completion_token
        else:
            messages = [
                {"role": "system", "content": collaborative_summary_system},
                {"role": "user", "content": collaborative_summary_user.render(doctor_info=doctors_answer_prompt, latest_info=self.current_report)}
            ]
            ans, prompt_token, completion_token = self.invoke(messages)
            action = self.extract_action_response(ans)
            self.current_report = ans['answer'] + "\n" + ans['report']
            return action, ans, messages, prompt_token, completion_token

    def generate_doctors_initial_prompt(self, doctor_responses: List[Dict]) -> str:
        responses = ""
        for idx in range(len(doctor_responses)):
            i = doctor_responses[idx]
            prompt = initial_doctor_for_revise.render(diagnosis=i['logit'], analysis=i['analysis'])
            if idx < len(doctor_responses) - 1:
                response = "Doctor " + str(idx) + "'s statement is as following:\n" + prompt + "\n"
            else:
                response = "Doctor " + str(idx) + "'s statement is as following:\n" + prompt
            responses = responses + response
        return responses

    def revise_logits(self, doctor_initial_responses) -> tuple[Dict[str, str], int, int]:
        doctors_answer_prompt = self.generate_doctors_initial_prompt(doctor_initial_responses)
        messages = [
            {"role": "system", "content": revise_logits_system},
            {"role": "user", "content": revise_logits_user.render(doctor_info=doctors_answer_prompt, latest_info=self.current_report)}
        ]
        ans, prompt_token, completion_token = self.invoke(messages)
        return ans, prompt_token, completion_token

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                stream=False
            )
            content = response.choices[0].message.content
            prompt_token = response.usage.prompt_tokens
            completion_token = response.usage.completion_tokens
            ans = extract_and_parse_json(content)
        except Exception as e:
            raise e
        end = time.time()
        print(f"MetaAgent time cost: {end - start:.2f}s.")
        return ans, prompt_token, completion_token


class Collaboration():
    def __init__(self, leader_agent: LeaderAgent, doctor_agents: List[DoctorAgent], leader_opinion: str, leader_report: str, save_dir: str, doctor_num: int = 3, max_round: int = 3):
        self.leader_opinion = leader_opinion
        self.leader_report = leader_report
        self.doctor_responses = []
        self.doctor_num = doctor_num
        self.leader_agent = leader_agent
        self.doctor_agents = doctor_agents
        self.max_round = max_round
        self.now_round = 0
        self.save_dir = save_dir

    def collaborate(self):
        assert len(self.doctor_responses) == 0 and self.now_round == 0
        prompt_tokens = 0
        completion_tokens = 0

        while self.now_round < self.max_round:
            self.now_round += 1
            self.doctor_responses = []

            def process_doctoragents(doctor_agent, i):
                response, messages, prompt_token, completion_token = doctor_agent.collaboration(self.leader_opinion, self.leader_report)
                self.doctor_responses.append(response)
                json.dump(response, open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review.json", "w"))
                json.dump(messages, open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review_messages.json", "w"))
                with open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review_userprompt.txt", "w") as f:
                    f.write(messages[1]["content"])
                return prompt_token, completion_token

            with ThreadPoolExecutor(max_workers=self.doctor_num) as executor:
                futures = [executor.submit(process_doctoragents, doctor_agent, i) for i, doctor_agent in enumerate(self.doctor_agents)]

                for future in futures:
                    prompt_token, completion_token = future.result()
                    prompt_tokens += prompt_token
                    completion_tokens += completion_token

                # for i, doctor in enumerate(self.doctor_agents):
                #     response, messages, prompt_token, completion_token = doctor.collaboration(
                #         self.leader_opinion, self.leader_report)
                #     self.doctor_responses.append(response)
                #     json.dump(response, open(
                #         f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review.json", "w"))
                #     json.dump(messages, open(
                #         f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review_messages.json", "w"))
                #     with open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review_userprompt.txt", "w") as f:
                #         f.write(messages[1]["content"])
                #     prompt_tokens += prompt_token
                #     completion_tokens += completion_token

            action, summary_content, messages, prompt_token, completion_token = self.leader_agent.summary(self.doctor_responses)
            json.dump(summary_content, open(f"{self.save_dir}/leader_round{self.now_round}_summary.json", "w"))
            json.dump(messages, open(f"{self.save_dir}/leader_round{self.now_round}_summary_messages.json", "w"))
            prompt_tokens += prompt_token
            completion_tokens += completion_token
            self.leader_opinion = summary_content["answer"]
            self.leader_report = summary_content["report"]

            if action == 0:
                break
            print("continue to next round")

        if self.now_round == self.max_round:
            print("No agreement has been reached")
        self.now_round = 0
        return self.leader_report, self.now_round, prompt_tokens, completion_tokens


def extract_and_parse_json(text: str):
    pattern_backticks = r'```json(.*?)```'
    match = re.search(pattern_backticks, text, re.DOTALL)
    if match:
        json_string = match.group(1).strip().replace("\n", "")
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(json_string)
            raise ValueError("Invalid JSON content found in match1.")

    match = re.search(pattern_backticks, text.strip() + "\"]}```", re.DOTALL)
    if match:
        json_string = match.group(1).strip().replace("\n", "")
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(json_string)
            raise ValueError("Invalid JSON content found in match 2.")

    pattern_json_object = r'\{.*?\}'
    match = re.search(pattern_json_object, text, re.DOTALL)
    if match:
        json_string = match.group(0).strip().replace("\n", "")
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(json_string)
            raise ValueError("Invalid JSON content found in match 3.")

    raise ValueError("No valid JSON content found.")