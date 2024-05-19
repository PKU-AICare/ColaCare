from liquid import Template

keywords_system = '''
You are a helpful medical expert and good at using tools. Your task is to searching for information about the question by calling the following tool: {
    "name": "search_engine",
    "description": "Search for information that will aid in determining a response to the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "search keywords"
            }
        },
        "required": ["input"]
    }
}. Your responses will be used for research purposes only, so please have a definite answer.
'''

keywords_user = Template('''
Here is the question:
{{question}}

Please respond with only the information you want to search without any other information, and format that in the input field as {"input": Str(Your search query)} which can be loaded by python json library, multiple search queries are allowed as a list of strings:
''')

medrag_system = '''You are a helpful medical expert. I will provide you with electronic health data of a End-Stage Renal Disease (ESRD) patient and some analysis results from our model, with the analysis results including mortality risk and feature attention. Your task is to analyze if the model's analysis results are reasonable using the relevant documents. Your responses will be used for research purposes only, so please have a definite answer.'''

medrag_user = Template('''
Here are the relevant documents:
{{context}}

Here are the analysis results:
{{hcontext}}

Note that the analysis results from the AI model are not all correct. Please analyze by following the steps below based on relevant documents:
1. Which of the relevant documents support the AI model's analysis results? Which do not? Please directly cite sentences or paragraphs from the documents' content in your explanation.
2. Do you think whether the AI model's analysis results are reasonable? The prediction of mortality risk is higher or lower than the actual risk? Please provide your prediction of mortality risk as a number between 0 and 1.
Please think step-by-step and generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction:
''')

ensemble_system = '''You are a helpful medical expert. I will provide you with electronic health data of a End-Stage Renal Disease (ESRD) patient and some analysis results from several models. Every model's analysis results include mortality risk and feature attention. Your task is to ensemble analysis results of all models and provide your prediction result of mortality risk using the relevant documents. Your responses will be used for research purposes only, so please have a definite answer.'''

ensemble_user = Template('''
Here are the relevant documents:
{{context}}

Here are the analysis results:
{{hcontext}}

Note that the analysis results from the AI models are not all correct. Please analyze by following the steps below based on relevant documents:
1. Please list the documents supporting the analysis results for each AI model separately and directly cite sentences or paragraphs from the documents' content in your explanation.
2. Please analyze whether the analysis results of each AI model are relatively reasonable, high, or low.
3. Please Analyze the patient's information, relevant documents, and the analysis results of all AI models, and provide your prediction result of mortality risk as a number between 0 and 1.
Please think step-by-step and generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction:
''')