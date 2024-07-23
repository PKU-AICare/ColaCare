from liquid import Template

initial_doctor_summary = Template("""He thinks the probability that the patient is ill is: {{diagnosis}}.
He first briefly analyzed the patient's basic condition: {{analysis}}
Then, he listed some evidence as the basis for my diagnosis of this patient: {{evidence}}"""
)
doctor_vote_agree_certain = Template("""Regarding the summary report of the leader expert, I AGREE WITH the leader, with GREAT CONFIDENCE.
The reason to support my statement is announced below:
{{reason}}
"""
)
doctor_vote_agree_uncertain = Template("""Regarding the summary report of the leader expert, I AGREE WITH the leader, but NOT VERY SURE.
The reason to support my statement is announced below:
{{reason}}
"""
)
doctor_vote_disagree_certain = Template("""Regarding the summary report of the leader expert, I DISAGREE WITH the leader, with GREAT CONFIDENCE.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I think my opinions and evidence are very important for accurate judgments. I hope the experts can consider them more.
"""
)
doctor_vote_disagree_uncertain = Template("""Regarding the summary report of the leader expert, I DISAGREE WITH the leader, but NOT VERY SURE.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I have doubts about the summary report of the leader expert, but I am not sure about my own opinions either. I have listed some evidence as a supplementary reference.
""")


action_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several general doctors have made analysis and judgments on a patient's condition. Your task is to judge whether everyone has reached a consensus on the diagnosis of the patient based on the analysis statements of each doctor."""

action_user = Template("""In response to the patient's summary report, several general doctors put forward their own opinions and reasons. 
In each doctor's statement, they first expressed whether they agreed with the previous summary report and gave the degree of confidence in their own judgment. Then, they further elaborated on their views by stating reasons and listing relevant literature. 
The following are their opinions:
{{opinions}}

Next, you need to judge whether the next round of discussion is needed based on each doctor's statement. Considering the following four cases:
(1) If all doctors agree with the previous summary report, there is no need to continue the discussion. 
(2) If some doctors disagree with the previous report, but they are not confident in their judgment and have not listed convincing evidence, there is no need to continue the discussion. 
(3) If some doctors strongly oppose the previous report and you think their evidence is worth discussing, we need to continue the discussion. 
(4) If most doctors disagree with the previous report, we need to continue the discussion.

Please output the following two contents in JSON format: 
(1) Whether to continue the discussion or not.
(2) Explain the reasons for your judgment.

Here are two examples of the format you should output:
{"Action": "Stop the discussion", "Reason": "All doctors agree with the previous summary report"}.
{"Action": "Continue the discussion", "Reason": "Exist a few experts strongly disagree with  the previous summary report"}.

Please respond in JSON format without any additional content:
"""
)

initial_summary_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several general doctors have made analysis and judgments on a patient's condition.Your task is to analyze the rationality of each doctor's opinion, summarize the opinions you think are reasonable to obtain a summary report for the patient and give your judgement of whether the patient has the disease."""

initial_summary_user = Template("""First, please read the patient's basic information carefully, as follows:
{{patient_info}}

Then, All doctors then made a diagnosis as to whether the patient had the disease and gave their reasons.
In each doctor's statement, they first analyzed the important features they believed to be related to the disease, then made a prediction about the probability of the patient getting the disease. Lastly, they collected and sorted out relevant literature as evidence for their judgment. 
The following are their opinions:
{{doctor_info}}

You need to read all doctors' opinion carefully and analyze whether their opinions make sense and whether they are helpful in diagnosing the patient's condition.
Next, please write a summary report including the following:
(1) An direct statement : in your opinion, whether the patient has the disease or not
(2) A summary of the patient's condition and the characteristics of the patient that you think are worthy of attention.
(3) List of supporting evidence: [Doctor 0's opinion or relevant literature retrieved, Doctor 1's opinion or relevant literature retrieved, ...].

Here are two examples of the format you should output:
{"Answer":"In my opinion, the patient is more probably to have the disease", "Report": "...", "Evidences": ["Doctor 0's ... ", "Doctor 1's ..."]}.
{"Answer":"In my opinion, the patient is more probably to not have the disease", "Report": "...", "Evidences": ["Doctor 0's ... ", "Doctor 1's ..."]}.

Please respond in JSON format without any additional content:"""
)

collaborative_summary_system = """You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several general doctors have made analysis and judgments on a patient's condition.Your task is to analyze the rationality of each doctor's opinion and summarize the opinions you think are reasonable to obtain a summary report for the patient."""

collaborative_summary_user = Template("""In the previous discussion, you took into account the opinions of all the doctors and obtained a summary report about the patient, which is listed as followed:
{{latest_info}}

Then, all the doctors offer new perspectives and opinions on your summary report.
In each doctor's statement, they first expressed whether they agreed with your statetement in the previous summary report and gave the degree of confidence in their own judgment. Then, they further elaborated on their views by stating reasons and listing relevant literature. 
The following are their opinions:
{{doctor_info}}

You need to think one by one, consider ALL the doctors' new ideas and modify your original summary report. 
(1) If a doctor expresses strong opposition to your previous statement, you need to focus on his reasons and arguments and think carefully about whether you need to think twice of whether the patient has the disease, and correspondingly, modify your summary report. 
(2) If a doctor expresses opposition, but he also has some doubts about his own opinion, you need to consider his opinion, of course, you can stick to your original opinion. 
(3) If a doctor expresses an agreement, then you do not need to modify your original summary based on his opinion.

Please output the following three contents in JSON format: 
(1) Your statement of whether the patient has the disease or not.
(2) Your revised summary report.
(3) Your reasons for revision in json format. 
The format of the reason for revision is: which doctor's opinion or relevant literature did you refer to, and which original opinion did you modify.

Here are two examples of the format you should output:
{"Answer": "In my opinion, the patient is more probably to have the disease", "Report": "...", "Reasons": ["reason1 ...", "reason2 ..."]}
{"Answer": "In my opinion, the patient is more probably to not have the disease", "Report": "...", "Reasons": ["reason1 ...", "reason2 ..."]}

Please respond in JSON format without any additional content:"""
)

doctor_analysis_system = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health record data for a patient staying in the ICU, as well as some AI model analyses for this patient, including death risk prediction results and feature importance weight. Please note that the greater the feature importance weight, the greater the impact of that feature on the patient's outcome. When the patient's risk of death is low, this impact is positive; conversely, when the risk is high, the impact is negative. I will also retrieve relevant medical knowledge based on the AI model's analysis results and provide it to you. Please summarize the patient's condition based on the above information and generate an analytical report.'''

doctor_analysis_user = Template('''Here is the relevant medical knowledge:
{{context}}

Here is the healthcare context, including the patient's basic information and analysis results of AI models:
{{hcontext}}

You need to analyze the patient's condition based on the above information and generate an analytical report. Please output the following contents in JSON format:
(1) The patient's death risk from AI model, just respond with a floating number.
(2) Your analysis of the patient's condition.
(3) Choose reasonable evidence from the medical knowledge I provide to support your analysis.

Here is an example of the format you should output:
{"Logit": "0.73", "Analysis": "Your analysis of the patient's condition.", "Evidences": ["Evidence 1 ...", "Evidence 2 ..."]}

Please respond in JSON format without any additional content:
''')

doctor_collaboration_system = '''You are an experienced medical expert participating in a consultation with several other medical experts for an ICU patient. The consultation organizer has generated a summary report based on all experts' analyses of the patient. Please provide your opinion on this report.'''

doctor_collaboration_user = Template('''Here is the relevant medical knowledge:
{{context}}

Here is the summary report generated by the consultation organizer:
{{report}}

You need to provide your opinion on the summary report generated by the consultation organizer. Please output the following two contents in JSON format:
(1) Your opinion on the summary report, respond with "agree" or "disagree".
(2) The confidence level of your opinion, respond with a number between 0 and 1.
(3) The reason for your opinion.
(4) If you disagree, the evidence you used to support your opinion. Please choose from the medical knowledge I provide as your evidence.

Here are two examples of the format you should output:
{"Answer": "agree", "Confidence": 0.8, "Reason": "The reason for your opinion."}
{"Answer": "disagree", "Confidence": 0.2, "Reason": "The reason for your opinion.", "Evidences": ["Evidence 1 ...", "Evidence 2 ..."]}

Please respond in JSON format without any additional content:
''')

##########################################################################################

keywords_system = '''
You are a helpful medical expert with extensive medical knowledge. Your task is to searching for information about the context by calling the following tool: {
    "name": "search_engine",
    "description": "Search for information that will aid in determining a response to the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "search keywords"
            }
        },
        "required": ["keywords"]
    }
}. You should summarize the context concisely into several keywords, ensuring that all the essential information is covered comprehensively. Your summary should be informative and beneficial for predicting in-hospital mortality.
'''

keywords_user = Template('''
Here is the context:
{{context}}

Multiple search keywords are allowed as a list of strings and format that in the keywords field as {"keywords": List(Your search queries)} which can be loaded by python json library. Your summary should be informative and beneficial for in-hospital mortality prediction task.

Here is an example of the format you should output:
{"keywords": ["End-Stage Renal Disease", "Hypertensive kidney damage", "Low Albumin level", "Carbon dioxide binding power", "Low Diastolic blood pressure", "Low Blood chlorine"]}

Please respond with only the information you want to search in JSON format without any additional information:
''')

retcare_system = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from our model, with the analysis results including mortality risk and feature importance weight. Your task is to analyze if the model's analysis results are reasonable using the relevant documents. Your analysis should be based on the relevant documents, and do not include any unsupported conclusions.'''

retcare_user = Template('''
Here are the relevant documents:
{{context}}

Here is the healthcare context, including the patient's basic information, analysis results of AI models and similar patients' information:
{{hcontext}}

Note that the analysis results from the AI model are not all correct. Please analyze by following the steps below based on relevant documents:
1. Which of the relevant documents support the AI model's analysis results? Which do not? Please directly cite sentences or paragraphs from the documents' content in your explanation.
2. Do you think whether the AI model's analysis results are reasonable? The prediction of mortality risk is higher or lower than the actual risk? Please provide your analysis based on the relevant documents, and disjudge the analysis results of the AI model if necessary.
3. Please provide your prediction of mortality risk as a number between 0 and 1.

Please think step-by-step and analyze the results based on relevant documents. Do not include any unsupported conclusions. Generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction:
''')

ensemble_evaluate_system_esrd = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and provide your prediction result of mortality risk using the relevant documents.'''

ensemble_evaluate_system_icu = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and provide your prediction result of mortality risk using the relevant documents.'''

ensemble_evaluate_user = Template('''
Here are the relevant documents:
{{context}}

Here is the healthcare context, including the patient's basic information, analysis results of AI models and similar patients' information:
{{hcontext}}

Note that the analysis results from the AI model are not all correct. Please think step-by-step and analyze the results based on relevant documents. Generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction, and explanation is your analysis following the template below:
## Summary
    Please describe your task, summarize the patient's basic information and health status, and restate the AI model's prediction results along with the feature importance and partial statistic information.
## Documents Analysis
    Please analyze the important features identified by the models, determine whether the identified features are reasonable. If they are reasonable, provide and cite relevant literature, include quotations from the sources, and explain the reasoning. If the features are not reasonable, provide and rank important features in your analysis, cite relevant documents and explain. Ensure that the features you identify are present in the dataset.
## Prediction Analysis
    Please evaluate the AI models' prediction results: too low, too high, or reasonable? If it's not reasonable, please provide your own prediction results, represented as a float number between 0 and 1.
''')

ensemble_select_system_esrd = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and select one result of models as your prediction result based on the relevant documents.'''

ensemble_select_system_covid = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of a patient admitted with a diagnosis of COVID-19 or suspected COVID-19 infection and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and select one of them as your final result based on the relevant documents.'''

ensemble_select_system_icu = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of a patient in Intensive Care Unit (ICU) and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and select one result of models as your prediction result based on the relevant documents.'''

ensemble_select_user = Template('''
Here are the relevant documents:
{{context}}

Here is the healthcare context, including the patient's basic information, analysis results of AI models and similar patients' information:
{{hcontext}}

Note that the analysis results from the AI model are not all correct. Please think step-by-step and analyze the results based on relevant documents. Generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction, and explanation is your analysis following the template below:
## Summary
    Please describe your task, summarize the patient's basic information and health status, and restate the AI model's prediction results along with the feature importance and partial statistic information.
## Documents Analysis
    Please analyze the important features identified by the models, determine whether the identified features are reasonable. If they are reasonable, provide and cite relevant literature, include quotations from the sources, and explain the reasoning. If the features are not reasonable, provide and rank important features in your analysis, cite relevant documents and explain. Ensure that the features you identify are present in the dataset.
## Prediction Analysis
    Please evaluate the AI models' prediction results: too low, too high, or reasonable? Choose one of them as your result, represented as a float number between 0 and 1.
''')