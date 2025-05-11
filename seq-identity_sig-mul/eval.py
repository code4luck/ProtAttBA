import os
import torch
from lit_model import LitModel
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import seed_everything
from utils.data_utils import get_AB_pair_data, get_s1131_data
from dataset import SeqDataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_locate", type=str, default="./model/esm2_650m")

    parser.add_argument(
        "--ckpt_locate",
        type=str,
        default="./test_ckpt/S1131/esm2_650m_S1131-val_pearson_corr_lr-3e-05_loss-mse.ckpt",
    )
    parser.add_argument(
        "--preds_path",
        type=str,
        default="./test_preds/S1131",
    )
    # dataset
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--filt_path", type=str, default="./data/csv_S1131/S1131_test.csv"
    )
    parser.add_argument("--max_length", type=int, default=None)

    # model
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--num_heads", type=int, default=4)  # 1280 // 4 = 320
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")  # 不指定则为False

    # Trainer
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    seed_everything(args.seed)
    return args


class EvalDataset(Dataset):
    def __init__(self, args, data):
        super(EvalDataset, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.model_locate = args.model_locate
        self.max_length = args.max_length
        self.truncation = True if self.max_length else False

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_locate)
        self.dataset = self.get_dataset(data)

    def get_dataset(self, data):
        (
            wt_ab_seqs,
            wt_ag_seqs,
            mt_ab_seqs,
            mt_ag_seqs,
            labels,
        ) = data
        dataset = SeqDataset(
            wt_ab_seqs=wt_ab_seqs,
            mt_ab_seqs=mt_ab_seqs,
            wt_ag_seqs=wt_ag_seqs,
            mt_ag_seqs=mt_ag_seqs,
            labels=labels,
        )
        return dataset

    def collator_fn(self, batch):
        wt_ab_seqs, mt_ab_seqs, wt_ag_seqs, mt_ag_seqs, labels = zip(*batch)

        wt_ab_inputs = self.tokenizer(
            wt_ab_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )
        mt_ab_inputs = self.tokenizer(
            mt_ab_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )
        wt_ag_inputs = self.tokenizer(
            wt_ag_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )
        mt_ag_inputs = self.tokenizer(
            mt_ag_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )
        labels = torch.tensor(labels, dtype=torch.float32)  # [B]

        batch = {
            "wt_ab_inputs_ids": wt_ab_inputs["input_ids"],
            "wt_ab_inputs_mask": wt_ab_inputs["attention_mask"],
            "mut_ab_inputs_ids": mt_ab_inputs["input_ids"],
            "mt_ab_inputs_mask": mt_ab_inputs["attention_mask"],
            "wt_ag_inputs_ids": wt_ag_inputs["input_ids"],
            "wt_ag_inputs_mask": wt_ag_inputs["attention_mask"],
            "mut_ag_inputs_ids": mt_ag_inputs["input_ids"],
            "mt_ag_inputs_mask": mt_ag_inputs["attention_mask"],
            "labels": labels,
        }
        return batch

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
        )


def load_eval_dataset(filt_path):
    if "S1131" in filt_path:
        (
            ab_hl_chains,
            ag_ab_chains,
            mt_ab_hl_chains,
            mt_ag_ab_chains,
            labels,
        ) = get_s1131_data(filt_path)
    elif "AB" in filt_path:
        (
            ab_hl_chains,
            ag_ab_chains,
            mt_ab_hl_chains,
            mt_ag_ab_chains,
            labels,
        ) = get_AB_pair_data(filt_path)
    else:
        raise ValueError(f"Invalid dataset name: {filt_path}")
    data = (
        ab_hl_chains,
        ag_ab_chains,
        mt_ab_hl_chains,
        mt_ag_ab_chains,
        labels,
    )
    return data


def eval(args):
    model = LitModel.load_from_checkpoint(args.ckpt_locate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.model
    model.eval()
    model.to(device)

    # dataset
    data = load_eval_dataset(args.filt_path)
    dataset = EvalDataset(args, data)
    data_loader = dataset.get_dataloader()

    # predict
    preds = []
    ground_truth = []
    with torch.no_grad():
        print("=" * 30, "start eval", "=" * 30)
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(
                wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
                wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
                mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
                mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
                wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
                wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
                mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
                mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
            )
            preds.extend(outputs.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
        print("=" * 30, "end eval", "=" * 30)
    # save preds
    print("=" * 30, "save preds", "=" * 30)
    preds = np.array(preds)
    ground_truth = np.array(ground_truth)
    df = pd.DataFrame(
        {"ground_truth": ground_truth, "preds": preds},
        columns=["ground_truth", "preds"],
    )
    if not os.path.exists(args.preds_path):
        os.makedirs(args.preds_path)
    file_name = args.ckpt_locate.split("/")[-1].split(".")[0]
    save_path = os.path.join(args.preds_path, f"{file_name}.csv")
    df.to_csv(save_path, index=False)


def main():
    args = parse_args()
    eval(args)


if __name__ == "__main__":
    main()
