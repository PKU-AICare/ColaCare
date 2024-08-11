import os
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.metrics_utils import get_binary_metrics, check_metric_is_better, run_bootstrap
from hparams import config
from fusion import Fusion

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.empty_cache()


class MyDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        super().__init__()
        self.ehr_embeddings = pd.read_pickle(os.path.join(data_path, f"{mode}_ehr.pkl"))
        self.text_embeddings = pd.read_pickle(os.path.join(data_path, f"{mode}_text.pkl"))
        self.y = pd.read_pickle(os.path.join(data_path, f"{mode}_y.pkl"))
        self.pids = pd.read_pickle(os.path.join(data_path, f"{mode}_pids.pkl"))
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        ehr_embedding = self.ehr_embeddings[index]
        text_embedding = self.text_embeddings[index]
        y = self.y[index]
        pid = self.pids[index]
        return ehr_embedding, text_embedding, y, pid


class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size, data_path):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = MyDataset(data_path, mode="train")
        self.val_dataset = MyDataset(data_path, mode='val')
        self.test_dataset = MyDataset(data_path, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


class Pipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.main_metric = config["main_metric"]
        self.output_dim = 1
        self.model = Fusion(ehr_embed_dim=config["ehr_embed_dim"], ehr_num=config["doctor_num"], text_embed_dim=config["text_embed_dim"], merge_embed_dim=config["merge_embed_dim"], output_dim=self.output_dim)
        self.loss_fn = nn.BCELoss()

        self.cur_best_performance = {} # val set
        self.test_performance = {} # test set

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_outputs = {}

    def forward(self, batch):
        ehr, text, y, pid = batch
        y_hat = self.model(ehr, text).to(text.device)
        return y_hat

    def _get_loss(self, batch):
        ehr, text, y, pid = batch
        y_hat = self(batch)
        y = y.to(y_hat.dtype)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, _ = self._get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ehr, text, y, pid = batch
        loss, y_hat = self._get_loss(batch)
        self.log("val_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        self.log("val_loss_epoch", loss)

        metrics = get_binary_metrics(y_pred, y_true)
        for k, v in metrics.items(): self.log(k, v)

        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, main_score, self.main_metric):
            self.cur_best_performance = metrics
            for k, v in metrics.items(): self.log("best_"+k, v)
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, batch_idx):
        ehr, text, y, pid = batch
        loss, y_hat = self._get_loss(batch)
        self.log("test_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'test_loss': loss}
        self.test_step_outputs.append(outs)
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.test_step_outputs]).detach().cpu()
        loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean().detach().cpu()
        self.log("test_loss_epoch", loss)

        test_performance = get_binary_metrics(y_pred, y_true)
        for k, v in test_performance.items(): self.log("test_"+k, v)

        self.test_outputs = {'y_pred': y_pred, 'y_true': y_true, 'test_loss': loss}
        self.test_step_outputs.clear()

        self.test_performance = test_performance
        return test_performance

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def run_experiment(config):
    # data
    dm = MyDataModule(batch_size=config["batch_size"], data_path=f"ehr_datasets/{config['ehr_dataset_name']}/processed/fold_1/fusion")

    # logger
    logger = CSVLogger(save_dir="logs", name=f"fusion/{config['ehr_dataset_name']}/{'-'.join(config['ehr_model_names'])}_{config['llm_name']}_{config['corpus_name']}", flush_logs_every_n_steps=1)

    # EarlyStop and checkpoint callback
    early_stopping_callback = EarlyStopping(monitor="auroc", patience=config["patience"], mode="max")
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="auroc", mode="max")

    L.seed_everything(42) # seed for reproducibility

    # train/val/test
    pipeline = Pipeline(config)
    trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)

    # Load best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print("best_model_path:", best_model_path)
    pipeline = Pipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs


if __name__ == "__main__":
    for hparam in config:
        print(hparam)
        perf, outs = run_experiment(hparam)
        metrics = run_bootstrap(outs['y_pred'], outs['y_true'])
        metrics = {k: f"{v['mean']*100:.2f} ± {v['std']*100:.2f}" for k, v in metrics.items()}
        print(pd.DataFrame(metrics, index=[0]))