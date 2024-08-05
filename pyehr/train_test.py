import os
import pandas as pd

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from configs.hparams import hparams
from ehrdatasets.loader.datamodule import EhrDataModule
from ehrdatasets.loader.load_los_info import get_los_info
from metrics.bootstrap import run_bootstrap
from pipelines import DlPipeline, MlPipeline


def run_ml_experiment(config):
    los_config = get_los_info(f'/home/wangzixiang/ColaCare/ehr_datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'/home/wangzixiang/ColaCare/ehr_datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    logger = CSVLogger(save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)
    L.seed_everything(config["seed"]) # seed for reproducibility

    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=logger, num_sanity_val_steps=0)
    trainer.fit(pipeline, dm)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs


def run_dl_experiment(config):
    los_config = get_los_info(f'/home/wangzixiang/ColaCare/ehr_datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'/home/wangzixiang/ColaCare/ehr_datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    if "time_aware" in config and config["time_aware"] == True:
        checkpoint_filename+="-ta" # time-aware loss applied
    logger = CSVLogger(save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)

    # EarlyStop and checkpoint callback
    if config["task"] in ["outcome", "readmission", "multitask"]:
        early_stopping_callback = EarlyStopping(monitor="auprc", patience=config["patience"], mode="max",)
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="auprc", mode="max")
    elif config["task"] == "los":
        early_stopping_callback = EarlyStopping(monitor="mae", patience=config["patience"], mode="min",)
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="mae", mode="min")

    L.seed_everything(config["seed"]) # seed for reproducibility

    # train/val/test
    trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback], num_sanity_val_steps=0)
    # trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=1, logger=logger, callbacks=[early_stopping_callback, checkpoint_callback], num_sanity_val_steps=0, limit_train_batches=1, limit_val_batches=1, limit_test_batches=1)
    
    pipeline = DlPipeline(config)
    trainer.fit(pipeline, dm)
    best_model_path = checkpoint_callback.best_model_path

    # best_model_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/best.ckpt'
    pipeline = DlPipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs


if __name__ == "__main__":
    best_hparams = hparams # [TO-SPECIFY]
    all_df = pd.DataFrame()
    for i in range(len(best_hparams)):
        config = best_hparams[i]
        run_func = run_ml_experiment if config["model"] in ["LR", "XGBoost"] else run_dl_experiment
        seeds = [0]
        folds = [1]
        config["epochs"] = 50
        config["patience"] = 10
        for fold in folds:
            config["fold"] = fold
            best_metric = {}
            for seed in seeds:
                config["seed"] = seed
                save_dir = f'logs/test/{config["dataset"]}/{config["model"]}/fold_{fold}-seed_{seed}'
                os.makedirs(save_dir, exist_ok=True)
                print(config)
                perf, outs = run_func(config)
                metrics = run_bootstrap(outs['preds'], outs['labels'])
                metrics = {k: f"{v['mean']*100:.2f} ± {v['std']*100:.2f}" for k, v in metrics.items()}
                metrics_df = pd.DataFrame({'model': config["model"], **metrics}, index=[i])
                all_df = pd.concat([all_df, metrics_df], axis=0)
    all_df.to_csv(f'{config["dataset"]}_metrics.csv', index=False)
                # pd.to_pickle(perf, f'{save_dir}/perf2.pkl')
                # pd.to_pickle(outs, f'{save_dir}/outs2.pkl')
                # preds = outs['preds'].tolist()
                # pd.to_pickle(preds, f"ehrdatasets/{config['dataset']}/processed/fold_{config['fold']}/{config['model']}_seed{config['seed']}_output2.pkl")