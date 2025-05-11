

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW

import torch
from torch import nn
from model import SeqBindModel
from pytorch_lightning import LightningModule
from torchmetrics import (
    MeanSquaredError,
    SpearmanCorrCoef,
    PearsonCorrCoef,
    R2Score, 
    MeanAbsoluteError,
)
from collections import OrderedDict


class LitModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lr = args.lr

        self.model = SeqBindModel(args)

        self.valid_metrics = nn.ModuleDict(
            OrderedDict(
                [
                    ("val_mse", MeanSquaredError()),
                    ("val_spearman_corr", SpearmanCorrCoef()),
                    ("val_pearson_corr", PearsonCorrCoef()),
                    ("val_r2", R2Score()),
                    ("val_mae", MeanAbsoluteError())
                ]
            )
        )
        self.test_metrics = nn.ModuleDict(
            OrderedDict(
                [
                    ("test_mse", MeanSquaredError()),
                    ("test_spearman_corr", SpearmanCorrCoef()),
                    ("test_pearson_corr", PearsonCorrCoef()),
                    ("test_r2", R2Score()),
                    ("test_mae", MeanAbsoluteError())
                ]
            )
        )

        # self.optimizer = args.optimizer
        if args.loss == "mse":
            self.criterion = nn.MSELoss()
        elif args.loss == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError("Invalid loss function")

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx): 
        labels = batch["labels"]

        outs = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
        )
        loss = self.criterion(outs, labels)
        self.log("loss", loss.item(), on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outs = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
        )
        for name, metric in self.valid_metrics.items():
            metric(outs, labels)
            self.log(name, metric, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        outs = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
        )
        for name, metric in self.test_metrics.items():
            metric(outs, labels)
            self.log(name, metric, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        return optimizer
