from typing import List

from agent import *

class Collaboration():
    def __init__(self, leader_agent: LeaderAgent, doctor_agents: List[DoctorAgent], leader_opinion: str, leader_report: str, save_dir: str, doctor_num: int=3, max_round: int=3):
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

        while self.now_round < self.max_round:

            self.now_round += 1
            self.doctor_responses = []
            
            for i, doctor in enumerate(self.doctor_agents):
                response = doctor.collaboration(self.leader_opinion, self.leader_report)
                self.doctor_responses.append(response)
                json.dump(response, open(f"{self.save_dir}/doctor{i}_round{self.now_round}_analysis.json", "w"))

            action, content = self.leader_agent.next_action(self.doctor_responses)
            json.dump(content, open(f"{self.save_dir}/leader_round{self.now_round}_action.json", "w"))

            if action == 0:
                break

            print("continue to next round")
            summary_content = self.leader_agent.summary(self.doctor_responses)
            json.dump(summary_content, open(f"{self.save_dir}/leader_round{self.now_round}_summary.json", "w"))
            self.leader_opinion = summary_content["Answer"]
            self.leader_report = summary_content["Report"]
        
        if self.now_round == self.max_round:
            print("No agreement has been reached")
        
        self.now_round = 0
        
        return self.leader_report, self.now_round