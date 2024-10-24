# ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration

ColaCare is a framework that enhances Electronic Health Record (EHR) modeling by leveraging Large Language Model (LLM)-driven multi-agent collaboration. This project aims to improve clinical prediction tasks by combining the strengths of domain-specific expert models and general-purpose LLMs.

## Key Features

1. **EHR-Specific Multi-Agent Framework**: ColaCare integrates domain-specific expert models with LLMs to bridge the gap between structured EHR data analysis and text-based reasoning capabilities.
2. **Collaborative Decision-Making Process**: The framework employs doctor agents and a meta doctor agent to simulate real-world medical consultations, ensuring comprehensive analysis of patient health records.
3. **Medical Guideline Support**: ColaCare incorporates authoritative medical guidelines using retrieval-augmented generation (RAG) techniques, grounding predictions in current medical knowledge.
4. **Enhanced Predictive Performance**: Experiments on four real-world EHR datasets demonstrate ColaCare's superior performance in mortality prediction tasks compared to existing EHR-specific models.
5. **Interpretable Insights**: The framework generates detailed, interpretable reports for each prediction, potentially revolutionizing clinical decision support systems.

## Environmental Setups

- Create an environment `colacare` and activate it.

```bash
conda create -n colacare python=3.9
conda activate colacare
```

- Install the required packages.

```bash
pip install -r requirements.txt
```

## Usage

### Training EHR models in pyehr

- Run the following command to train the EHR models and obtain feature importance scores in `pyehr` directory.

```bash
cd pyehr
python train_test.py
python importance.py
```

> More details about the pre-processing and training steps can be found in the [`pyehr`](https://github.com/yhzhu99/pyehr) repository.

### Running LLM-based Multi-Agent Collaboration

- Write a configuration file for the multi-agent collaboration framework. 

```python
# hparams.py

mimic_config = {
    "retriever_name" : "MedCPT",
    "corpus_name" : "MSD",
    "llm_name" : "deepseek-chat",
    "epochs" : 50,
    "patience" : 10,
    "ehr_dataset_name" : 'mimic-iv',
    "ehr_dataset_dir" : './ehr_datasets/mimic-iv/processed/fold_1',
    "ehr_model_names" : ['AdaCare', 'MCGRU', 'RETAIN'], 
    "seeds": [0, 0, 0],
    "doctor_num" : 3,
    "max_round" : 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 32,
    "mode": "test"
}
```

- Run the following command to start the multi-agent collaboration framework.

```bash
python collaboration_pipeline.py
```

> The results can be found in the `response` directory.

### Training Fusion Network 

- Run the following command to train the fusion network.

```bash
python utils/process_output.py
python train_fusion.py
```

## Datasets and Tasks

### Datasets

ColaCare has been evaluated on the following datasets:
- MIMIC-IV
- CDSL (COVID-19 Data Saving Lives)
- ESRD (End-Stage Renal Disease)

### Tasks

ColaCare has been evaluated on the following tasks:
- In-hospital Mortality Prediction
- 30-day Readmission Prediction