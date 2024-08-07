from liquid import Template

revise_logits_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. After the discussion, you will get a consistent analysis report on the patient. Your task is to refer to each doctor's initial diagnosis and the conclusion of the analysis report to output your final prediction of the patient's mortality risk."""

revise_logits_user = Template("""Initially, all doctors made predictions about the mortality risk of the patient and explained their reasons, as shown below:
{{doctor_info}}

Then, in a collaborative consultation, you organized all doctors to get a consistent analysis report on the patient. The report includes:
1. Your judgment on mortality risk of the patient
2. Reasons and evidence to support your conclusion
As follows:
{{latest_info}}

You need to adjust the initial predictions of the patient's mortality risk by several doctors based on the results of the collaborative consultation. 

The specific explanation is as follows:
Doctor 0 thought the mortality risk of the patient was 0.24.
Doctor 1 thought the mortality risk of the patient was 0.50.
Doctor 2 ...

Based on the collaborative consultation, you considered the logits of Doctor 0, Doctor 1..., and the consistent analysis report. Finally, make a final prediction of the patient's mortality risk as refined final logit.

Please respond in the following JSON format:
{"Doctor 0's Logit": ..., "Doctor 1's Logit": ..., ..., "Final Logit": ...}

Respond in JSON format without any additional content:
""")

initial_doctor_for_revise = Template("""He thinks the mortality risk of the patient is: {{diagnosis}}.
Then, he briefly analyzed the patient's basic condition: {{analysis}}"""
)

initial_doctor_summary = Template("""He thinks the mortality risk of the patient is: {{diagnosis}}.
He first briefly analyzed the patient's basic condition: {{analysis}}
Then, he listed some evidence as the basis for my diagnosis of this patient: {{evidence}}"""
)

doctor_vote_agree_certain = Template("""Regarding the summary report of the leader expert, I agree with the leader, with high confidence.
The reason to support my statement is announced below:
{{reason}}"""
)

doctor_vote_agree_uncertain = Template("""Regarding the summary report of the leader expert, I agree with the leader, but am not very sure.
The reason to support my statement is announced below:
{{reason}}"""
)

doctor_vote_disagree_certain = Template("""Regarding the summary report of the leader expert, I disagree with the leader, with high confidence.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I think my opinions and evidence are very important for accurate judgments. I hope the leader expert can consider them more."""
)

doctor_vote_disagree_uncertain = Template("""Regarding the summary report of the leader expert, I disagree with the leader, but am not very sure.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I have doubts about the summary report of the leader expert, but I am not sure about my own opinions either. I have listed some evidence as a supplementary reference."""
)

action_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several general doctors have made analysis and judgments on a patient's condition. Your task is to judge whether everyone has reached a consensus on the diagnosis of the patient based on the analysis statements of each doctor."""

action_user = Template("""In response to the patient's summary report, several general doctors put forward their own opinions and reasons. 
In each general doctor's statement, they first expressed whether they agreed with the previous summary report and gave the degree of confidence in their own judgment. Then, they further elaborated on their views by stating reasons and listing relevant literature. 
The following are their opinions:
{{opinions}}

Next, you need to judge whether the next round of discussion is needed based on each general doctor's statement. Considering the following four cases:
1. If all doctors agree with the previous summary report, there is no need to continue the discussion.
2. If some doctors disagree with the previous report, but they are not confident in their judgment and have not listed convincing evidence, there is no need to continue the discussion.
3. If some doctors strongly oppose the previous report and you think their evidence is worth discussing, we need to continue the discussion.
4. If most doctors disagree with the previous report, we need to continue the discussion.

Please output the following two contents in JSON format:
1. Whether to continue the discussion or not.
2. Explain the reasons for your judgment.

Here are two examples of the format you should output:
{"Action": "Stop the discussion", "Reason": "All doctors agree with the previous summary report"}.
{"Action": "Continue the discussion", "Reason": "Exist a few experts strongly disagree with the previous summary report"}.

Respond in JSON format without any additional content:
"""
)

initial_summary_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several doctors have made analysis and judgments on a patient's condition. Your task is to analyze the rationality of each general doctor's opinion, summarize the opinions you think are reasonable to obtain a summary report for the patient and give your judgment on the patient's mortality risk."""

initial_summary_user = Template("""First, please read the patient's basic information carefully, as follows:
{{patient_info}}

Then, all doctors make a diagnosis on the patient's condition and give their reasons.
In each doctor's statement, they first analyze the important features they believe to be related to the patient's condition, then make a prediction about the mortality risk of the patient. Lastly, they collect and sort out relevant literature as evidence for their judgment.
The following are their opinions:
{{doctor_info}}

You need to read all doctors' opinions carefully and analyze whether their opinions make sense and whether they are helpful in diagnosing the patient's condition.
Next, please write a summary report including the following:
1. A direct statement: in your opinion, whether the mortality risk of the patient is higher or not.
2. A summary of the patient's condition and the characteristics of the patient that you think are worthy of attention.
3. List of supporting evidence: [Doctor 0's opinion or relevant literature retrieved, Doctor 1's opinion or relevant literature retrieved, ...].

Here are two examples of the format you should output:
{"Answer":"In my opinion, the patient has a high risk of mortality.", "Report": "...", "Evidences": ["Doctor 0's ... ", "Doctor 1's ..."]}.
{"Answer":"In my opinion, the patient has a low risk of mortality.", "Report": "...", "Evidences": ["Doctor 0's ... ", "Doctor 1's ..."]}.

Respond in JSON format without any additional content:
""")

collaborative_summary_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several doctors have made analysis and judgments on a patient's condition. Your task is to analyze the rationality of each doctor's opinion and summarize the opinions you think are reasonable to obtain a summary report for the patient."""

collaborative_summary_user = Template("""In the previous discussion, you took into account the opinions of all the doctors and obtained a summary report about the patient, which is listed as follows:
{{latest_info}}

Then, all the doctors offer new perspectives and opinions on your summary report.
In each doctor's statement, they first expressed whether they agreed with your statement in the previous summary report and gave the degree of confidence in their own judgment. Then, they further elaborated on their views by stating reasons and listing relevant literature.
The following are their opinions:
{{doctor_info}}

You need to consider all the doctors' new ideas and modify your original summary report.
1. If a doctor expresses strong opposition to your previous statement, you need to focus on his reasons and arguments and think carefully about whether you need to reconsider mortality risk of the patient and modify your summary report accordingly.
2. If a doctor expresses opposition but also has some doubts about his own opinion, you need to consider his opinion, but you can stick to your original opinion.
3. If a doctor expresses agreement, then you do not need to modify your original summary based on his opinion.

Please output the following three contents in JSON format:
1. Your statement of mortality risk of the patient.
2. Your revised summary report. Please create a concise and clear summary of the patient's health status. Your summary should be informative and beneficial for healthcare prediction tasks, such as in-hospital mortality prediction.
3. Your reasons for revision in JSON format. The format of the reason for revision is: which doctor's opinion or relevant literature did you refer to, and which original opinion did you modify.

Here is an example of the revised summary report:
35-year-old female was diagnosed with chronic glomerulonephritis five years ago. Chronic kidney disease progressed to end stage renal disease (ESRD) over last year. Client was started on automated peritoneal dialysis six months ago. She developed fever, vomiting, and abdominal pain 1 day ago. Current assessment: T 39.1 °C, (102.40 F), HR 104, RR 16, B/P 145/ 87 mmHg. Weight: 66.8 Kg/147 lbs. NKDA. Periumbilical tenderness, defense and rebound. Erythema and creamy, yellow exudate around peritoneal dialysis catheter exit site. Dialysate effluent is cloudy yellow. Peritoneal effluent culture was obtained. Labs sent. The patient's complex medical history and the presence of multiple comorbidities suggest a high risk for in-hospital mortality. Close monitoring and management of these conditions are critical for patient outcomes.

Here are two examples of the format you should output:
{"Answer": "In my opinion, the patient has a high risk of mortality.", "Report": "...", "Reasons": ["reason1 ...", "reason2 ..."]}
{"Answer": "In my opinion, the patient has a low risk of mortality.", "Report": "...", "Reasons": ["reason1 ...", "reason2 ..."]}

Respond in JSON format without any additional content:
""")

doctor_analysis_system = """You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health record data for a patient, as well as some AI model analyses for this patient, including death risk prediction results and feature importance weight. The greater the feature importance weight, the greater the impact of that feature on the patient's outcome. When the patient's risk of death is low, this impact is positive; conversely, when the risk is high, the impact is negative. I will also retrieve relevant medical knowledge based on the AI model's analysis results and provide it to you. Please summarize the patient's condition based on the above information and generate an analytical report."""

doctor_analysis_user = Template("""Here is the relevant medical knowledge:
{{context}}

Here is the healthcare context, including the patient's basic information and analysis results of AI models:
{{hcontext}}

You need to analyze the patient's condition based on the above information and generate an analytical report. Please output the following contents in JSON format:
1. The patient's death risk from AI model, just respond with a two-decimal number.
2. Your analysis of the patient's condition.
3. Choose reasonable evidence from the medical knowledge I provide to support your analysis.

Here is an example of the format you should output:
{"Logit": "0.73", "Analysis": "Your analysis of the patient's condition.", "Evidences": ["Evidence 1 ...", "Evidence 2 ..."]}

Respond in JSON format without any additional content:
""")

doctor_collaboration_system = """You are an experienced medical expert participating in a consultation with several other medical experts for a patient. The leader expert of this consultation has generated a summary report based on all experts' analyses of the patient. Please provide your viewpoint on his opinion."""

doctor_collaboration_user = Template("""Here is the relevant medical knowledge:
{{context}}

Here is your initial analysis of the patient:
{{analysis}}

Here is the opinion of the consultation organizer:
{{opinion}}

Here is the summary report generated by the consultation organizer:
{{report}}

You need to provide your opinion on the summary report generated by the consultation organizer. Please output the following two contents in JSON format:
1. Your viewpoint on the opinion of the consultation organizer, i.e., the patient's risk is either high or low, respond with "agree" or "disagree".
2. The confidence level of your opinion, respond with a number between 0 and 1.
3. The reason for your opinion.
4. If you disagree, the evidence you used to support your opinion. Please choose from the medical knowledge I provide as your evidence.

Here are two examples of the format you should output:
{"Answer": "agree", "Confidence": 0.8, "Reason": "The reason for your opinion."}
{"Answer": "disagree", "Confidence": 0.2, "Reason": "The reason for your opinion.", "Evidences": ["Evidence 1 ...", "Evidence 2 ..."]}

Respond in JSON format without any additional content:
""")