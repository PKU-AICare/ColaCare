import os

import pandas as pd
import shap
import torch
import numpy as np

from ehrdatasets.loader.datamodule import EhrDataModule
from ehrdatasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline
from configs.hparams import hparams

shap.initjs()
torch.set_grad_enabled(True)
BATCH_SIZE = 1024

def get_background_and_shap_variables(config):
    dm = EhrDataModule(f'ehrdatasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=BATCH_SIZE)
    x_bg, y_bg, lens_bg, pid_bg = (next(iter(dm.train_dataloader())))
    # only use the last visit to calculate the feature importance, and the shape is (batch_size, input_dim)
    x_bg = x_bg[:, -1, :].detach().cpu().numpy()
    x_shap, y_shap, lens_shap, pid_shap = (next(iter(dm.test_dataloader())))
    x_shap = x_shap[:, -1, :].detach().cpu().numpy()
    return x_bg, x_shap


def get_checkpoint_filename(ckpt_dir: str):
    filenames = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if len(filenames) == 1:
        filename = filenames[0]
    else:
        filenames.remove("best.ckpt") if "best.ckpt" in filenames else None
        filename = list(sorted(filenames, key=lambda x: int(x.split('-')[-1].split('.')[0][1:])))[-1]
    return filename


def get_feature_importance(config, x_bg, x_shap):
    # los_config = get_los_info(f'ehrdatasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    # config.update({"los_info": los_config})

    # checkpoint
    ckpt_dir = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints'
    filename = get_checkpoint_filename(ckpt_dir)
    ckpt_path = f'{ckpt_dir}/{filename}'

    if config["model"] in ["RF", "DT", "GBDT", "XGBoost", "CatBoost", "LR"]:
        pipeline = MlPipeline.load_from_checkpoint(ckpt_path, config=config)
    else:
        pipeline = DlPipeline.load_from_checkpoint(ckpt_path, map_location="cuda:1", config=config)

    def predict(x):
        output = pipeline.predict(x)
        return output

    if config["dataset"] == 'esrd':
        x_bg_kmeans = shap.kmeans(x_bg, k=2).data
    else:
        x_bg_kmeans = shap.kmeans(x_bg, k=32).data
    e = shap.KernelExplainer(predict, x_bg_kmeans)
    shap_values = e.shap_values(x_shap).squeeze()
    return shap_values

if __name__ == "__main__":
    best_hparams = hparams
    for i in range(len(best_hparams)):
        config = best_hparams[i]
        folds = [1]
        seeds = [0]
        for fold in folds:
            config["fold"] = fold
            save_dir = f"ehrdatasets/{config['dataset']}/processed/fold_{config['fold']}"
            os.makedirs(save_dir, exist_ok=True)
            for seed in seeds:
                config["seed"] = seed
                print(config)
                x_bg, x_shap = get_background_and_shap_variables(config)
                shap_values = get_feature_importance(config, x_bg, x_shap)
                pd.to_pickle(shap_values, f"{save_dir}/{config['model']}_seed{config['seed']}_shap.pkl")
                # shap_values = pd.read_pickle(f"{save_dir}/{config['model']}_seed{config['seed']}_shap.pkl")

                feature_names_url = f'{save_dir}/labtest_features.pkl'
                test_raw_x_url = f'{save_dir}/train_raw_x.pkl'
                feature_names = pd.read_pickle(feature_names_url)
                test_raw_x = pd.read_pickle(test_raw_x_url)

                if config["dataset"] in ['mimic-iv', 'mimic-iii']:
                    features = shap_values[:, config["demo_dim"] + 47:]
                    test_raw_x = np.array([x[-1] for x in test_raw_x])[:, config["demo_dim"] + 47:]
                else:
                    features = shap_values[:, config["demo_dim"]:]
                    test_raw_x = np.array([x[-1] for x in test_raw_x])[:, config["demo_dim"]:]
                print(features.shape, len(feature_names), test_raw_x.shape)

                all_features = []
                for feature_weight_item, raw_item in zip(features, test_raw_x):
                    last_feat_dict = {key: {'value': value, 'attention': attn} for key, value, attn in zip(feature_names, raw_item, feature_weight_item)}
                    last_feat_dict_sort = dict(sorted(last_feat_dict.items(), key=lambda x: abs(x[1]['attention']), reverse=True))
                    selected_features = [item for item in last_feat_dict_sort.items() if abs(item[1]['attention']) > 0.005][:3]
                    all_features.append(selected_features)
                pd.to_pickle(all_features, f'{save_dir}/{config["model"]}_seed{config["seed"]}_features.pkl')
                
                outs = pd.read_pickle(f"logs/test/{config['dataset']}/{config['model']}/fold_{fold}-seed_{config['seed']}/outs.pkl")
                preds = outs['preds'].tolist()
                pd.to_pickle(preds, f"{save_dir}/{config['model']}_seed{config['seed']}_output.pkl")
                embeddings = outs['embeddings']
                pd.to_pickle(embeddings, f"{save_dir}/{config['model']}_seed{config['seed']}_embeddings.pkl")