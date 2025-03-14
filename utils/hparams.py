cdsl_config = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",  # "PubMed", "MOC"
    "llm_name": "deepseek-chat",  # gpt-4o-mini
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "cdsl",
    "ehr_task": "outcome",
    "ehr_dataset_dir": "./ehr_datasets/cdsl/processed/fold_1",
    "ehr_model_names": ["AdaCare", "MCGRU", "RETAIN"],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,  # 1024 for GatorTron
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    "mode": "test",
}

cdsl_config_v2 = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",  # "PubMed", "MOC"
    "llm_name": "deepseek-chat",  # gpt-4o-mini
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "cdsl",
    "ehr_task": "outcome",
    "ehr_dataset_dir": "./ehr_datasets/cdsl/processed/fold_1",
    "ehr_model_names": ["AICare", "GRU", "M3Care"],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,  # 1024 for GatorTron
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    "mode": "test",
}

mimic_config = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",  # "PubMed", "MOC"
    "llm_name": "deepseek-chat",
    # "llm_name": "qwen-turbo",
    # "llm_name": "gpt-4o-mini",
    # "llm_name": "gpt-4o",
    # "llm_name": "doubao-pro-32k",
    # "llm_name": "Llama-3.1-405B",
    # "llm_name": "claude-3-5-sonnet-20241022",
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "mimic-iv",
    "ehr_task": "outcome", # "outcome" or "readmission"
    "ehr_dataset_dir": "./ehr_datasets/mimic-iv/processed/fold_1",
    "ehr_model_names": ["AdaCare", "MCGRU", "RETAIN"],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,  # 1024 for GatorTron
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    "mode": "test",
}

mimic_re_config = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",  # "PubMed", "MOC"
    "llm_name": "deepseek-chat",
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "mimic-iv",
    "ehr_task": "readmission", # "outcome" or "readmission"
    "ehr_dataset_dir": "./ehr_datasets/mimic-iv/processed/fold_1",
    "ehr_model_names": ["AdaCare", "MCGRU", "RETAIN"],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,  # 1024 for GatorTron
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    "mode": "val",
}

mimic_config_v2 = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",  # "PubMed", "MOC"
    "llm_name": "deepseek-chat",
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "mimic-iv",
    "ehr_task": "outcome",
    "ehr_dataset_dir": "./ehr_datasets/mimic-iv/processed/fold_1",
    "ehr_model_names": ["AICare", "GRU", "M3Care"],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,  # 1024 for GatorTron
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    "mode": "val",
}

mimic_re_config_v2 = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",  # "PubMed", "MOC"
    "llm_name": "deepseek-chat",
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "mimic-iv",
    "ehr_task": "readmission",
    "ehr_dataset_dir": "./ehr_datasets/mimic-iv/processed/fold_1",
    "ehr_model_names": ["AICare", "GRU", "M3Care"],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,  # 1024 for GatorTron
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    "mode": "val",
}

esrd_config = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",  # "PubMed", "MOC"
    "llm_name": "deepseek-chat",  # gpt-4o-mini
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "esrd",
    "ehr_task": "outcome",
    "ehr_dataset_dir": "./ehr_datasets/esrd/processed/fold_1",
    "ehr_model_names": ["concare", "RETAIN", "random"],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,  # 1024 for GatorTron
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    "mode": "test",
}

esrd_config_v2 = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",  # "PubMed", "MOC"
    "llm_name": "deepseek-chat",  # gpt-4o-mini
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "esrd",
    "ehr_task": "outcome",
    "ehr_dataset_dir": "./ehr_datasets/esrd/processed/fold_1",
    "ehr_model_names": ["AICare", "GRU", "M3Care"],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,  # 1024 for GatorTron
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    "mode": "val",
}

config = [
    cdsl_config,
    cdsl_config_v2,
    mimic_config,
    mimic_config_v2,
    mimic_re_config,
    mimic_re_config_v2,
    esrd_config,
    esrd_config_v2,
]
