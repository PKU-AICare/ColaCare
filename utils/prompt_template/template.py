from liquid import Template


## Phase 1
doctor_review_system = """You are an experienced doctor with extensive medical knowledge. I will provide you with multivariate time-series electronic health record for a patient, which is a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as sequences of numerical values for each feature. I will also provide some AI model analyses for this patient, including mortality risk prediction results and feature importance weights. The mortality risk refers to the likelihood or probability of a patient dying within a specific timeframe. The greater the feature importance weight, the greater the impact of that feature on the patient's outcome. When the patient's risk of mortality is low, this impact is positive; conversely, when the risk is high, the impact is negative. I will also retrieve relevant medical knowledge based on the AI model's analysis results and provide it to you. Please summarize the patient's condition based on multivariate time-series electronic health record (especially the values at final time-point), mortality risk, feature importance weights and medical knowledge, and generate an analytical review for the patient."""

doctor_review_user = Template("""Here is the relevant medical knowledge:
{{context}}

Here is the patient record, including the patient's basic information, multivariate time-series electronic health record and analysis results of AI models:
{{hcontext}}

You need to analyze the patient's overall condition based on the information above and generate an analytical review. Please output the following content in JSON format:
1. The patient's mortality risk from the AI model, just respond with a two-decimal number.
2. Your analysis of the patient's condition. Please pay great attention to the electronic health record at the final time-point and feature importance weights from AI models, and analyze based on them. Do not just list the patient's basic information or provide unsupported analysis. Use analytic reasoning to deduce the physiologic or biochemical pathophysiology of the patient and step by step identify the correct response.
3. Choose reasonable evidence from the medical knowledge I provide to support your analysis. Please list titles and important content from the medical knowledge represented as a List of strings. Do not just list the titles of the medical knowledge.

Here is an example of the format you should output:
{"logit": "0.73", "analysis": "Your analysis of the patient's condition.", "evidences": ["Evidence 1 ...", "Evidence 2 ..."]}

Respond in JSON format without any additional content:
""")


## Phase 2
meta_summary_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several doctors have made analysis and judgments on a patient's condition. Your task is to analyze the rationality of each doctor's opinion, summarize the opinions to obtain a synthesized report for the patient and give your judgment on the patient's mortality risk. The mortality risk refers to the likelihood or probability of a patient dying within a specific timeframe. The greater the mortality risk, the higher the likelihood of the patient dying."""

meta_summary_user = Template("""First, please read the patient's basic information carefully, as follows:
{{patient_info}}

Then, all doctors make a diagnosis on the patient's condition and give their reasons.
In each doctor's statement, they first analyze the important features they believe to be related to the patient's condition, then make a prediction about the mortality risk of the patient. Lastly, they collect and sort out relevant literature as evidence for their judgment.
The following are their opinions:
{{doctor_info}}

You need to read all doctors' opinions carefully and analyze whether their opinions make sense and whether they are helpful in diagnosing the patient's condition.
Next, please write a synthesized report including the following:
1. A direct statement: in your opinion, whether the mortality risk of the patient is higher or not.
2. A summary of the patient's condition and the characteristics of the patient that you think are worthy of attention. Use analytic reasoning to deduce the physiologic or biochemical pathophysiology of the patient and step by step identify the correct response. Please list important features and their impact on the patient's condition. But do not list all the features provided by the doctors.
3. A List of supporting evidence represented as a string. Please output detailed content from some of the evidence provided by the doctors or some analysis results of the doctors. Do not just list the doctor's name.

Here are two examples of the format you should output:
{"answer":"In my opinion, the patient has a high risk of mortality.", "report": "...", "evidences": ["...", "..."]}.
{"answer":"In my opinion, the patient has a low risk of mortality.", "report": "...", "evidences": ["...", "..."]}.

Respond in JSON format without any additional content:
""")


## Phase 3
doctor_collaboration_system = """You are an experienced medical expert participating in a consultation with several other medical doctors for a patient. The meta doctor of this consultation has generated a synthesized report based on all doctors' analysis of the patient. Please provide your viewpoint on his opinion."""

doctor_collaboration_user = Template("""Here is the relevant medical knowledge:
{{context}}

Here is your last analysis of the patient, which is not completely reasonable, and you may need to adjust it based on the meta doctor's opinion:
{{analysis}}

Here is the opinion of the meta doctor:
{{opinion}}

Here is the synthesized report generated by the meta doctor:
{{report}}

You need to consider the meta doctor's opinion carefully and provide your opinions on the synthesized report generated by the meta doctor. Please output your opinions including the following content in JSON format:
1. Your viewpoint on the opinion of the meta doctor, i.e., the patient's risk is either high or low, respond with "agree" or "disagree".
2. The confidence score of your opinion, respond with an integer between 1 and 3. The meaning of the confidence score is as follows: 
    3 for High - You are an expert in the subject area and have extensive knowledge in the medical domain. You are highly confident in your ability to provide an accurate and thorough assessment. Your evaluation is based on deep expertise and a comprehensive understanding of the work.
    2 for Moderate - You have a good understanding of the subject area and is familiar with the medical domain. You feel confident in your ability to accurately assess the quality and significance of the work. Your evaluation is based on a solid grasp of the content and context.
    1 for Low - You have some knowledge of the subject area and is somewhat familiar with the medical domain. You understand the main points but may lack depth in certain areas. You are reasonably confident in your assessment but acknowledges some limitations in your expertise.
3. The reason for your opinion. If you change your opinion, for example, you agree with the meta doctor's opinion which is different from your last analysis, please respond that you have changed in your response and provide detailed reasons for the change. You need to point out the parts where you got the wrong conclusion or the important parts you ignored in your last analysis, and the new key features that you think are important and the impact of these features on the patient's condition. Do not just repeat the meta doctor's opinions.
4. The evidence you use to support your opinion. Please choose from the relevant medical knowledge I provide as your evidence, and must output important content from the evidence.

Here are examples of the format you should output:
{"answer": "agree", "confidence": 3, "reason": "The reason for your opinion.", "evidences": ["Evidence 1 ...", "Evidence 2 ..."]},
{"answer": "disagree", "confidence": 1, "reason": "The reason for your opinion.", "evidences": ["Evidence 1 ...", "Evidence 2 ..."]}

Respond in JSON format without any additional content:
""")

action_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several doctors have made analysis and judgments on a patient's condition. Your task is to judge whether everyone has reached a consensus on the diagnosis of the patient based on the analysis statements of each doctor."""

action_user = Template("""In response to the patient's synthesized report, several doctors put forward their own opinions and reasons. 
In each doctor's statement, they first express whether they agree with the previous synthesized report and give the degree of confidence in their own judgment. Then, they further elaborate on their views by stating reasons and listing relevant literature. 
The following are their opinions:
{{opinions}}

Next, you need to judge whether the next round of discussion is needed based on each doctor's statement. Considering the following four cases:
1. If all doctors agree with the previous synthesized report, there is no need to continue the discussion.
2. If some doctors disagree with the previous report, but they are not confident in their judgment and have not listed convincing evidence, there is no need to continue the discussion.
3. If some doctors strongly oppose the previous report and you think their evidence is worth discussing, please continue the discussion.
4. If most doctors disagree with the previous report, please continue the discussion.

Please output the following content in JSON format:
1. Whether to continue the discussion or not.
2. Explain the reasons for your judgment.

Here are two examples of the format you should output:
{"action": "Stop the discussion", "reason": "All doctors agree with the previous synthesized report"}.
{"action": "Continue the discussion", "reason": "Exist a few doctors strongly disagree with the previous synthesized report"}.

Respond in JSON format without any additional content:
"""
)

collaborative_summary_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several doctors have made analysis and judgments on a patient's condition. Your task is to judge whether everyone has reached a consensus on the diagnosis of the patient based on the analysis statements of each doctor and then analyze the rationality of each doctor's opinion and summarize the opinions you think are reasonable to obtain a synthesized report for the patient."""

collaborative_summary_user = Template("""In the previous discussion, you took into account the opinions of all the doctors and obtained a synthesized report about the patient, which is listed as follows:
{{latest_info}}

In response to the patient's synthesized report, several doctors put forward their own opinions and reasons.
In each doctor's statement, they first express whether they agree with your statement in the previous synthesized report and give the confidence level on their own judgment. Then, they further elaborate on their views by stating reasons and listing relevant literatures.
The following are their opinions:
{{doctor_info}}

Now, you need to judge whether the next round of discussion is needed based on each doctor's statement. Considering the following four cases:
1. If all doctors agree with the previous synthesized report, there is no need to continue the discussion.
2. If some doctors disagree with the previous report, but they are not confident in their judgment and have not listed convincing evidence, there is no need to continue the discussion.
3. If some doctors strongly oppose the previous report and you think their evidence is worth discussing, please continue the discussion.
4. If most doctors disagree with the previous report, please continue the discussion.

If you think the discussion should continue, you need to analyze the rationality of each doctor's opinions and summarize the opinions you think are reasonable to obtain a synthesized report for the patient. You shold follow the following cases:
1. If a doctor expresses strong opposition to your previous statement, you need to focus on his reasons and arguments and think carefully about whether you need to reconsider mortality risk of the patient and modify your synthesized report accordingly.
2. If a doctor expresses opposition but also has some doubts about his own opinion, you need to consider his opinion, but you can stick to your original opinion.
3. If a doctor expresses agreement, then you do not need to modify your original synthesized report based on his opinion.

Please output the following three contents in JSON format:
1. Whether to continue the discussion or not. Please respond with `Yes` or `No`.
2. Your statement of mortality risk of the patient, i.e., whether the mortality risk of the patient is higher or not.
3. Your revised synthesized report. Please create a concise and clear summary of the patient's health status. Your summary should be informative and beneficial for healthcare prediction tasks, such as in-hospital mortality prediction. Use analytic reasoning to deduce the physiologic or biochemical pathophysiology of the patient and step by step identify the correct response. Please list important features one by one and their impact on the patient's condition. But do not list all the features provided by the doctors.
4. Your reasons for revision. The format of the reason for revision is: which doctor's opinion or relevant literature you refer to, and which original opinions you modify. Please output detailed content, don't just list the doctor's name.

Here are examples of the revised synthesized report, please follow this format:
1. 35-year-old female was diagnosed with chronic glomerulonephritis five years ago. Chronic kidney disease progressed to end stage renal disease (ESRD) over last year. Client was started on automated peritoneal dialysis six months ago. She developed fever, vomiting, and abdominal pain 1 day ago. Current assessment: T 39.1 °C, (102.40 F), HR 104, RR 16, B/P 145/ 87 mmHg. Weight: 66.8 Kg/147 lbs. NKDA. Periumbilical tenderness, defense and rebound. Erythema and creamy, yellow exudate around peritoneal dialysis catheter exit site. Dialysate effluent is cloudy yellow. Peritoneal effluent culture was obtained. Labs sent. The patient's complex medical history and the presence of multiple comorbidities suggest a high risk for in-hospital mortality. Close monitoring and management of these conditions are critical for patient outcomes.
2. The patient is a 75-year-old female with End-Stage Renal Disease (ESRD) due to Diabetic Nephropathy, along with co-morbidities of Diabetes and Heart failure. Key features contributing to her high mortality risk include: 
1) Elevated creatinine levels (167.0 to 545.0 \u00b5mol/L) and blood urea nitrogen (BUN) levels (9.3 to 26.1 mmol/L), indicating severe renal dysfunction. 
2) Fluctuating blood potassium levels, with a recent dangerously low value of 2.04 mmol/L, which can lead to cardiac arrhythmias. 
3) Low blood calcium levels (1.73 to 2.29 mmol/L), common in chronic kidney disease, contributing to bone disease and cardiovascular complications. 
4) Decreasing albumin levels (44.0 to 22.3 g/L), indicating malnutrition or inflammation. 
5) Elevated hypersensitive C-reactive protein (CRP) levels (117.01 mg/L), suggesting significant inflammation. 
6) Variable blood glucose levels, with a recent spike to 22.1 mmol/L, indicating poor glycemic control. 
7) Variable systolic blood pressure, generally elevated, which can exacerbate heart failure and renal disease. 
These factors collectively indicate a complex and severe clinical condition with multiple interacting pathophysiological processes contributing to the patient's high mortality risk.

Here are two examples of the format you should output:
{"action": "No", "answer": "In my opinion, the patient has a high risk of mortality.", "report": "...", "reasons": ["reason1 ...", "reason2 ..."]}
{"action": "Yes", "answer": "In my opinion, the patient has a low risk of mortality.", "report": "...", "reasons": ["reason1 ...", "reason2 ..."]}

Respond in JSON format without any additional content:
""")

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
{"doctor0_logit": ..., "doctor1_logit": ..., "doctor2_logit": ..., "final_logit": ...}

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
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}"""
)

doctor_vote_agree_uncertain = Template("""Regarding the summary report of the leader expert, I agree with the leader, but am not very sure.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}"""
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

doctor_high_confidence = Template("""Regarding the summary report of the leader expert, I {{opinion}} with the leader.
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I am an expert in the subject area and have extensive knowledge in the medical domain. I am highly confident in my ability to provide an accurate and thorough assessment. My evaluation is based on deep expertise and a comprehensive understanding of the work.
""")

doctor_moderate_confidence = Template("""Regarding the summary report of the leader expert, I {{opinion}} with the leader.
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I have a good understanding of the subject area and is familiar with the medical domain. I feel confident in your ability to accurately assess the quality and significance of the work. My evaluation is based on a solid grasp of the content and context.
""")

doctor_low_confidence = Template("""Regarding the summary report of the leader expert, I {{opinion}} with the leader.
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I have some knowledge of the subject area and is somewhat familiar with the medical domain. I understand the main points but may lack depth in certain areas. I am reasonably confident in my assessment but acknowledges some limitations in my expertise.
""")        