import torch, os, sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
from litmodel import LitModel
from dataset import WrapperDataset
from utils.data_split import get_K_fold_generator
from utils.test_callback import MetricsToFileCallback
from utils.common import read_s645_csv, get_ab645_pair_data
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_locate", type=str, default="./model/esm2_650m")

    # dataset
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_path", type=str, default="./data/csv/AB645.csv")
    parser.add_argument("--data_name", type=str, default="AB645")

    # model
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true") 

    # Trainer
    parser.add_argument("--n_fold", type=int, default=10)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--monitor", type=str, default="val_pearson_corr")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=5.0)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="value")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--precision", type=str, default="32")

    args = parser.parse_args()

    seed_everything(args.seed)
    return args


def main():
    args = parse_args()
    # mutil-cross-validation
    file_path = args.data_path

    ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels = (
        get_ab645_pair_data(file_path)
    )

    assert (
        len(ab_hl_chains)
        == len(ag_ab_chains)
        == len(mt_ab_hl_chains)
        == len(mt_ag_ab_chains)
        == len(labels)
    ), "data length not consistent"

    # get K_fold generator
    len_data = np.arange(len(labels))
    generator = get_K_fold_generator(len_data, n_fold=args.n_fold, seed=args.seed)
    print("args: ", args)
    model_name = args.model_locate.split("/")[-1]
    # kfolds train
    for fold, (train_index, test_index) in enumerate(generator):
        dataset = WrapperDataset(
            args=args,
            train_idxes=train_index,
            test_idxes=test_index,
            wt_ab_seqs=ab_hl_chains,
            mt_ab_seqs=mt_ab_hl_chains,
            wt_ag_seqs=ag_ab_chains,
            mt_ag_seqs=mt_ag_ab_chains,
            labels=labels,
        )
        print(f"cur {fold} test_index ", test_index)

        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()
        test_loader = dataset.get_test_loader()

        monitor = args.monitor
        data_name = args.data_name

        model = LitModel(args)
        mode = "max" if "corr" in monitor else "min"
        ckpt_name = (
            f"{model_name}_{data_name}-fold-{fold}-{monitor}_lr-{args.lr}_loss-{args.loss}_index.ckpt"
        )
        model_checkpoint = ModelCheckpoint(
            f"checkpoints/{data_name}/{fold}",  # !
            monitor=monitor,
            mode=mode,
            filename=ckpt_name,
            verbose=True,
        )

        early_stop = EarlyStopping(
            monitor=monitor, mode=mode, patience=args.patience, verbose=True
        )
        metrics2file = MetricsToFileCallback(
            data_name=data_name,
            fold=fold,
            monitor=monitor,
            model_path=args.model_locate,
            loss_fn=args.loss,
            lr=args.lr,
        )
        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            strategy=args.strategy,
            num_nodes=args.num_nodes,
            max_epochs=args.max_epochs,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            deterministic=True,
            # precision=args.precision,
            callbacks=[model_checkpoint, early_stop, metrics2file],
        )
        trainer.fit(model, train_loader, val_loader)
        model = LitModel.load_from_checkpoint(model_checkpoint.best_model_path)
        trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
