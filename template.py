from liquid import Template

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

ensemble_evaluate_system = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and provide your prediction result of mortality risk using the relevant documents.'''

ensemble_evaluate_user = Template('''
Here are the relevant documents:
{{context}}

Here is the healthcare context, including the patient's basic information, analysis results of AI models and similar patients' information:
{{hcontext}}

Note that the analysis results from the AI models are not all correct. Please analyze by following the steps below based on relevant documents:
1. Please list the documents supporting the analysis results for each AI model separately and directly cite sentences or paragraphs from the documents' content in your explanation.
2. Please analyze whether the analysis results of each AI model are relatively reasonable, high, or low.
3. Please analyze the patient's information, based on relevant documents and the analysis results of all AI models, and provide your prediction result of mortality risk as a number between 0 and 1. Do not give the mean value of all models' analysis results.

Please think step-by-step and analyze the results based on relevant documents. Do not include any unsupported conclusions. Generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction:
''')

ensemble_select_system = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of a End-Stage Renal Disease (ESRD) patient and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and select one result of models as your prediction result based on the relevant documents.'''

ensemble_select_user = Template('''
Here are the relevant documents:
{{context}}

Here is the healthcare context, including the patient's basic information, analysis results of AI models and similar patients' information:
{{hcontext}}

Note that the analysis results from the AI models are not all correct. Please analyze by following the steps below based on relevant documents:
1. Please list the documents supporting the analysis results for each AI model separately and directly cite sentences or paragraphs from the documents' content in your explanation.
2. Please analyze whether the analysis results of each AI model are relatively reasonable, high, or low.
3. Please analyze the patient's information, relevant documents, and the analysis results of all AI models, and choose one of the results as the final output. If you think the results of all models are unreasonable, please provide your prediction result of mortality risk as a number between 0 and 1.

Please think step-by-step and analyze the results based on relevant documents. Do not include any unsupported conclusions. Generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction:
''')