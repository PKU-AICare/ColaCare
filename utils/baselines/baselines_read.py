from liquid import Template

zeroshot = {
    "system": """
        You are an experienced doctor with extensive medical knowledge. I will provide you with multivariate time-series electronic health record for a patient, which is a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as sequences of numerical values for each feature and the 30-day readmission prediction result refers to the likelihood of readmission after 30 days since discharge. The greater the feature importance weight, the greater the impact of that feature on the patient's readmission likelihood. When the patient's likelihood of readmission is low, this impact is positive; conversely, when the likelihood is high, the impact is negative. Please summarize the patient's condition based on multivariate time-series electronic health record, readmission likelihood and generate an analytical review."""
    ,
    "user": Template("""
        The patient's health records are provided in the following format: {{hcontext}}.
        Please assess the provided medical data and analyze the health records to determine the likelihood of the patient's 30-day readmission rate. 
        Give your judgement with the 30-day readmission rate (a floating-point number between 0 and 1) and the analysis report based on the provided data.
        If the patient is likely to survive, but you claim the patient death rate is high, you would be penalized.
        Respond in JSON format as following, without any additional content: {"logits": "...", "reason": "My analysis of the patient's condition is ..."}
    """),
}

fewshot = {
    "system": """
        You are an experienced doctor with extensive medical knowledge. I will provide you with multivariate time-series electronic health record for a patient, which is a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as sequences of numerical values for each feature and the 30-day readmission prediction result refers to the likelihood of readmission after 30 days since discharge. The greater the feature importance weight, the greater the impact of that feature on the patient's readmission likelihood. When the patient's likelihood of readmission is low, this impact is positive; conversely, when the likelihood is high, the impact is negative. Please summarize the patient's condition based on multivariate time-series electronic health record, readmission likelihood and generate an analytical review."""
    ,
    "user": Template("""   
        Please assess the provided medical data and analyze the health records to determine the likelihood of the patient's 30-day readmission rate. 
        Give your judgement with the 30-day readmission rate (a floating-point number between 0 and 1) and the analysis report based on the provided data.
        If the patient is not likely to be readmitted to the hospital, but you claim the patient 30-day readmission rate is high, you would be penalized.
        
        Two examples are provided for you to follow:
        1. The patient's health records is : This female patient, aged 47, is an patient in Intensive Care Unit (ICU).\n\nHere is complete medical information from multiple visits of a patient, with each feature within this data as a string of values separated by commas.\n- Diastolic blood pressure: \"88.0, 88.0, 64.0......"\n- Fraction inspired oxygen: \"0.5, 0.5, 0.5......"\n- Blood glucose: \"131.0, 196.0, 196.0......"\n- Heart Rate: \"84.0, 86.0, 86.0......"
           The following judge is: "Logits": "0.15", "Reason": "My analysis of the patient's condition is as follows: The patient's diastolic blood pressure shows some variability but remains within a range that is generally considered safe for most patients. The fraction inspired oxygen is consistently at 0.5, which is typical for ICU patients requiring supplemental oxygen. Blood glucose levels are elevated but not critically so, and the heart rate is within a normal range......"
        2. The patient's health records is : This male patient, aged 69, is an patient in Intensive Care Unit (ICU).\n\nHere is complete medical information from multiple visits of a patient, with each feature within this data as a string of values separated by commas.\n- Diastolic blood pressure: \"47.0, 77.0, 70.0......\"\n- Fraction inspired oxygen: \"0.5, 1.0, 0.5......\"\n- Blood glucose: \"217.0, 176.0, 177.0, ......\"\n- Heart Rate: \"105.0, 100.0, 94.0......\"
           The following judge is: "Logits": "0.85", "Reason": "My analysis of the patient's condition is as follows: The patient, a 69-year-old male in the ICU, exhibits several concerning trends. Diastolic blood pressure shows a significant drop from 77.0 to 36.0, indicating potential cardiovascular instability......"        
        
        The certain patient's health records are provided in the following format: {{hcontext}}.
        
        Respond in JSON format as following, without any additional content: {"logits": "...", "reason": "My analysis of the patient's condition is ..."}
    """)
}

sc = {
    "system": """
        You are an experienced doctor with extensive medical knowledge. I will provide you with multivariate time-series electronic health record for a patient, which is a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as sequences of numerical values for each feature and the 30-day readmission prediction result refers to the likelihood of readmission after 30 days since discharge. The greater the feature importance weight, the greater the impact of that feature on the patient's readmission likelihood. When the patient's likelihood of readmission is low, this impact is positive; conversely, when the likelihood is high, the impact is negative. Please summarize the patient's condition based on multivariate time-series electronic health record, readmission likelihood and generate an analytical review."""
    ,
    "user": Template("""   
        Please assess the provided medical data and analyze the health records to determine the likelihood of the patient's 30-day readmission rate. 
        Give your judgement with the 30-day readmission rate (a floating-point number between 0 and 1) and the analysis report based on the provided data.
        If the patient is not likely to be readmitted to the hospital, but you claim the patient 30-day readmission rate is high, you would be penalized.
         
        The certain patient's health records are provided in the following format: {{hcontext}}. 
        First generate three different reports to obtain more diagnostic information.
        Then, analyze these three diagnostic reports and give the final probability of 30-day readmission rate and the corresponding analysis report.
        
        Respond in JSON format as following, without any additional content: {"reason_1": "I think the patient is not likely to be readmitted to the hospital in 30 days because...", "reason_2": "I think the patient is likely to be readmitted to the hospital in 30 days because...", "reason_3": "I think the patient is likely to ...", "reason":"...", "logits": "..."}
    """)
}

specialist = [
    {
    "system": """
        You are an experienced doctor with extensive medical knowledge. I will provide you with multivariate time-series electronic health record for a patient, which is a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as sequences of numerical values for each feature and the 30-day readmission prediction result refers to the likelihood of readmission after 30 days since discharge. The greater the feature importance weight, the greater the impact of that feature on the patient's readmission likelihood. When the patient's likelihood of readmission is low, this impact is positive; conversely, when the likelihood is high, the impact is negative. Please summarize the patient's condition based on multivariate time-series electronic health record, readmission likelihood and generate an analytical review."""
    ,
    "user": Template("""
        The patient's health records are provided in the following format: {{hcontext}}.
        Please assess the provided medical data and analyze the health records to determine the likelihood of the patient not surviving their hospital stay. 
        Give your judgement with the death probability (a floating-point number between 0 and 1) and the analysis report based on the provided data.
        If the patient is likely to survive, but you claim the patient death rate is high, you would be penalized.
        Respond in JSON format as following, without any additional content: {"logits": "...", "reason": "My analysis of the patient's condition is ..."}
    """)
    },
    {
    "system": """
        You are an experienced doctor with extensive medical knowledge. I will provide you with electronic health record data for a patient from multiple visits of a patient, each characterized by a fixed number of features. 
    """,
    "user": Template("""
        The patient's health records are provided in the following format: {{hcontext}}.
        Please assess the provided medical data and analyze the health records to determine the likelihood of the patient not surviving their hospital stay. 
        Give your judgement with the death probability (a floating-point number between 0 and 1) and the analysis report based on the provided data.
        If the patient is likely to survive, but you claim the patient death rate is high, you would be penalized.
        Respond in JSON format as following, without any additional content: {"logits": "...", "reason": "My analysis of the patient's condition is ..."}
    """)
    }
]


debate_collaboration = {
    "system": "You are an experienced doctor with extensive medical knowledge. Please refer to other doctors' opinions and modify your original diagnosis.",
    "user": Template("""
        In the previous round, you think the 30-day readmission rate of the patient is {{logit}}. And your initial analysis of the patient is {{analysis}}.
        Your previous judgement may be not completely reasonable, and you may need to adjust it based on other doctor's opinion:
        {{reports}}

        You need to consider other doctors' opinion carefully and Please output your opinions including the following content in JSON format:
        1. Your revised logits about patient's 30-day readmission rate, respond with a floating-point number between 0 and 1.
        2. Your revised summary report, considering your original statement and other doctors' opinions.

        Here are two examples of the format you should output:
        {"revised_report":"I think the patient is likely to be readmitted to the hospital in 30 days, because ...","revised_logits": "0.8"}
        {"revised_report":"I think the patient is not likely to be readmitted to the hospital in 30 days, because ...","revised_logits": "0.1"}
        
        Respond in JSON format without any additional content:
    """)
}

specialist_collaboration = {
    "system": """
        You are an authoritative expert in the medical field. You are organizing a collaborative consultation. Now several doctors have made analysis and judgments on a patient's condition. Your task is to analyze the rationality of each doctor's opinion and summarize the opinions you think are reasonable to obtain a synthesized report for the patient.
    """,
    "user": Template("""
        all the doctors offer their final opinions about the patient, including the death rate and the corresponding report.
        The following are their opinions:
        {{reports}}

        You need to consider other doctors' opinion carefully and Please output your opinions including the following content in JSON format:
        1. Your revised logits about patient's 30-day readmission rate, respond with a floating-point number between 0 and 1.
        2. Your revised summary report, considering your original statement and other doctors' opinions.

        Here are two examples of the format you should output:
        {"revised_report":"Consider all doctors' opinion, I think the patient is likely to be readmitted to the hospital in 30 days, because ...","revised_logits": "0.8"}
        {"revised_report":"Consider all doctors' opinion, I think the patient is not likely to be readmitted to the hospital in 30 days, because ...","revised_logits": "0.1"}
        
        Respond in JSON format without any additional content:
    """)
}

confidence_collaboration = {
    "system": "You are an experienced doctor with extensive medical knowledge. Please refer to other doctors' opinions and modify your original diagnosis.",
    "user": Template("""
        In the previous round, you think the 30-days readmission rate of the patient is {{logit}}. And your initial analysis of the patient is {{analysis}}.
        Your previous judgement may be not completely reasonable, and you may need to adjust it based on other doctor's opinion:
        {{reports}}

        You need to consider other doctors' opinion carefully and . Please output your opinions including the following content in JSON format:
        1. Your revised logits about patient's 30-day readmission rate, respond with a floating-point number between 0 and 1.
        2. Your revised summary report, considering your original statement and other doctors' opinions.
        3. The confidence level of your opinion, respond with a number between 0 and 1.
        Remember: if you are not confident but report a high confidence level, you will be penalized!

        Here are two examples of the format you should output:
        {"revised_report":"I think the patient is likely to be readmitted to the hospital in 30 days, because ...","revised_logits": "0.8", "confidence": 0.8}
        {"revised_report":"I think the patient is not likely to be readmitted to the hospital in 30 days, because ...","revised_logits": "0.1", "confidence": 0.3}
        
        Respond in JSON format without any additional content:
    """)
}
            