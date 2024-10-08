medical_char = {
    "Cl": "Chlorine",
    "CO2CP": "CO2CP",
    "WBC": "WBC",
    "Hb": "HGB",
    "Urea": "Urea",
    "Ca": "Calcium",
    "K": "Potassium",
    "Na": "Sodium",
    "Scr": "SCR",
    "P": "PHOS",
    "Albumin": "Albumin",
    "hs-CRP": "hs-CRP",
    "Glucose": "Glucose",
    "Appetite": "Food intake",
    "Weight": "Weight",
    "SBP": "Systolic pressure",
    "DBP": "Diastolic pressure",
}

medical_name = {
    "Cl": "Blood chlorine",
    "CO2CP": "Carbon dioxide binding power",
    "WBC": "White blood cells",
    "Hb": "Hemoglobin",
    "Urea": "Urea",
    "Ca": "Blood calcium",
    "K": "Blood potassium",
    "Na": "Blood sodium",
    "Scr": "Creatinine",
    "P": "Blood phosphorus",
    "Albumin": "Albumin",
    "hs-CRP": "Hypersensitive C reactive protein",
    "Glucose": "Blood glucose",
    "Appetite": "Food intake (with water)",
    "Weight": "Weight",
    "SBP": "Systolic blood pressure",
    "DBP": "Diastolic blood pressure",
}

medical_unit = {
    "Cl": "mmol/L",
    "CO2CP": "mmol/L",
    "WBC": "×10^9/L",
    "Hb": "g/L",
    "Urea": "mmol/L",
    "Ca": "mmol/L",
    "K": "mmol/L",
    "Na": "mmol/L",
    "Scr": "μmol/L",
    "P": "mmol/L",
    "Albumin": "g/L",
    "hs-CRP": "mg/L",
    "Glucose": "mmol/L",
    "Appetite": "g",
    "Weight": "kg",
    "SBP": "mmHg",
    "DBP": "mmHg",
}

medical_show = {
    'Cl': {'step': 1, 'range': [82, 120]},
    'CO2CP': {'step': 0.5, 'range': [10.5, 39.5]},
    'WBC': {'step': 0.5, 'range': [0, 24]},
    'Hb': {'step': 2, 'range': [44, 180]},
    'Urea': {'step': 1, 'range': [3, 44]},
    'Ca': {'step': 0.05, 'range': [1, 3.9]},
    'K': {'step': 0.1, 'range': [1.9, 7.9]},
    'Na': {'step': 0.5, 'range': [121, 150]},
    'Scr': {'step': 20, 'range': [0, 1920]},
    'P': {'step': 0.05, 'range': [0.5, 3]},
    'Albumin': {'step': 1, 'range': [10, 57]},
    'hs-CRP': {'step': 2, 'range': [0, 118]},
    'Glucose': {'step': 0.5, 'range': [1, 24.5]},
    'Weight': {'step': 1, 'range': [29, 108]},
    'SBP': {'step': 2, 'range': [40, 200]},
    'DBP': {'step': 2, 'range': [20, 136]},
}

medical_standard = {
    "Cl": [96, 106],
    "CO2CP": [20, 29],
    "WBC": [3.5, 9.5],
    "Hb": [115, 150],
    "Urea": [3.1, 8.0],
    "Ca": [2.25, 2.75],
    "K": [3.5, 5.5],
    "Na": [135, 145],
    "Scr": [62, 115],
    "P": [1.1, 1.3],
    "Albumin": [40, 55],
    "hs-CRP": [0.5, 10],
    "Glucose": [3.9, 6.1],
    "SBP": [100, 120],
    "DBP": [60, 80],
}

medical_standard_for_esrd = {
    "Cl": "higher than 96 mmol/L",
    "CO2CP": "higher than 25 mmol/L",
    "Hb": "higher than 114 g/L",
    "Urea": "higher than 20 mmol/L",
    "Ca": "higher than 2.5 mmol/L",
    "K": "higher than 4 mmol/L",
    "Na": "higher than 135.5 mmol/L",
    "Scr": "higher than 900 μmol/L",
    "P": "higher than 1.5 mmol/L",
    "Albumin": "higher than 32 g/L",
    "hs-CRP": "lower than 16 mg/L",
    "Glucose": "lower than 6 mmol/L",
    "Weight": "higher than 59 kg",
    "SBP": "higher than 130 mmHg",
    "DBP": "higher than 70 mmHg",
}

medical_keys = [
    "Cl",
    "CO2CP",
    "WBC",
    "Hb",
    "Urea",
    "Ca",
    "K",
    "Na",
    "Scr",
    "P",
    "Albumin",
    "hs-CRP",
    "Glucose",
    "Appetite",
    "Weight",
    "SBP",
    "DBP",
]

disease_chinese = {
    "diabetes": '糖尿病',
    "heart_failure": '心脏衰竭',
    "lung_infect": '肺部感染',
    "chd": '冠心病',
    "mi": '心梗',
    "ci": '脑梗',
    "ch": '脑出血',
    "amputation": '截肢',
}

disease_english = {
    "Diabetes": 'Diabetes',
    "Heart_failure": 'Heart failure',
    "Lung_infect": 'Lung infection',
    "CHD": 'Coronary heart disease',
    "MI": 'Myocardial infarction',
    "CI": 'Cerebral infarction',
    "CH": 'Cerebral hemorrhage',
    "Amputation": 'Amputation'
}

original_disease = {
    '慢性肾小球肾炎': 'Chronic glomerulonephritis',
    '慢性肾小球性肾炎': 'Chronic glomerulonephritis',
    '不明': 'unspecified',
    '糖尿病肾病': 'Diabetic Nephropathy',
    '高血压肾损害': 'Hypertensive kidney damage',
    '慢性肾盂肾炎': 'Chronic pyelonephritis',
    '良性动脉硬化': 'Benign arteriosclerosis',
    '慢性间质性肾炎': 'Chronic interstitial nephritis',
    '常染色体显性多囊肾病': 'Autosomal dominant polycystic kidney disease',
    '梗阻性肾病': 'Obstructive nephropathy',
    '原发性小血管炎肾损害': 'Primary small vessel vasculitis renal damage',
    '高血压肾病': 'Hypertensive nephropathy',
    '肾动脉硬化': 'Renal arteriosclerosis',
    '缺血性肾病': 'Ischemic nephropathy',
    "多发性骨髓瘤肾损害": 'Renal damage in multiple myeloma',
    "多囊肾": 'Polycystic kidney',
    "肾移植后肾炎": 'Post-transplant nephritis',
    "系统性红斑狼疮性肾炎": 'Systemic lupus erythematosus nephritis',
    "包虫病": 'Inclusion disease',
    "淀粉样变性肾病": 'Amyloidosis nephropathy',
    "痛风慢性间质性肾炎": 'Gout chronic interstitial nephritis',
    "未知": 'unknown',
    "null": 'null',
    "其他": 'Other',
    "过敏性紫癜性肾炎": 'Allergic purpura nephritis',
    "DN": 'DN',
    "IGA肾病": 'IGA nephropathy',
    "局灶增生性肾小球肾炎伴肾小管间质":
        'Focal proliferative glomerulonephritis with tubulointerstitial',
    "肾病综合症伴膜性增殖性IGA肾病":
        'tubulointerstitial Nephrotic syndrome with membranoproliferative IGA nephropathy ',
    "肾炎": 'nephritis',
    '高血压肾损害肾小球肾炎?': 'Hypertensive renal damage glomerulonephritis?',
    "心肾综合征": 'Cardiorenal syndrome',
    "糖尿病肾病心衰": 'Diabetic nephropathy heart failure'
}

####################################################### mimic-iv #######################################################

mimic_range = {
    "Diastolic blood pressure": "less than 80 mmHg",
    "Fraction inspired oxygen": "more than 0.21",
    "Glucose": "70 mg/dL - 100 mg/dL",
    "Heart Rate": "60 bpm - 100 bpm",
    "Mean blood pressure": "less than 100 mmHg",
    "Oxygen saturation": "95 % - 100 %",
    "Respiratory rate": "15 breaths per minute - 18 breaths per minute",
    "Systolic blood pressure": "less than 120 mmHg",
    "Temperature": "36.1 degrees Celsius - 37.2 degrees Celsius",
    "pH": "7.35 - 7.45"
}

mimic_unit = {
    "Diastolic blood pressure": "mmHg",
    "Glucose": "mg/dL",
    "Heart Rate": "bpm",
    "Height": "cm",
    "Mean blood pressure": "mmHg",
    "Oxygen saturation": "%",
    "Respiratory rate": "breaths per minute",
    "Systolic blood pressure": "mmHg",
    "Temperature": "degrees Celsius",
    "Weight": "kg",
}