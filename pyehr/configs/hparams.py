hparams = [
    {
        "model": "LR",
        "calib": False,
        "dataset": "cdsl",
        "task": "outcome",
        "max_depth": 5,
        "n_estimators": 50,
        "learning_rate": 0.1,
        "batch_size": 81920,
        "main_metric": "auprc",
    },
    {
        "model": "MCGRU",
        "calib": False,
        "dataset": "cdsl",
        "task": "outcome",
        "main_metric": "auprc",
        "epochs": 50,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
    },
]