
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from transformers import AutoTokenizer


class SeqDataset(Dataset):
    def __init__(
        self,
        wt_ab_seqs: List[str],
        mt_ab_seqs: List[str],
        wt_ag_seqs: List[str],
        mt_ag_seqs: List[str],
        labels: List[float],
    ):
        self.wt_ab_seqs = wt_ab_seqs
        self.mt_ab_seqs = mt_ab_seqs
        self.wt_ag_seqs = wt_ag_seqs
        self.mt_ag_seqs = mt_ag_seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.wt_ab_seqs[idx],
            self.mt_ab_seqs[idx],
            self.wt_ag_seqs[idx],
            self.mt_ag_seqs[idx],
            self.labels[idx],
        )


class WrapperDataset(Dataset):
    def __init__(
        self,
        args,
        train_idxes,
        test_idxes,
        wt_ab_seqs,
        mt_ab_seqs,
        wt_ag_seqs,
        mt_ag_seqs,
        labels,
    ):
        super(WrapperDataset, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.model_locate = args.model_locate

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_locate)
        self.train_dataset, self.val_dataset = self.get_dataset(
            train_idxes,
            test_idxes,
            wt_ab_seqs,
            mt_ab_seqs,
            wt_ag_seqs,
            mt_ag_seqs,
            labels,
        )

    def get_dataset(
        self,
        train_idxes,
        test_idxes,
        wt_ab_seqs,
        mt_ab_seqs,
        wt_ag_seqs,
        mt_ag_seqs,
        labels,
    ):
        train_wt_ab_seqs = [wt_ab_seqs[i] for i in train_idxes]
        train_mt_ab_seqs = [mt_ab_seqs[i] for i in train_idxes]
        train_wt_ag_seqs = [wt_ag_seqs[i] for i in train_idxes]
        train_mt_ag_seqs = [mt_ag_seqs[i] for i in train_idxes]
        train_labels = [labels[i] for i in train_idxes]

        test_wt_ab_seqs = [wt_ab_seqs[i] for i in test_idxes]
        test_mt_ab_seqs = [mt_ab_seqs[i] for i in test_idxes]
        test_wt_ag_seqs = [wt_ag_seqs[i] for i in test_idxes]
        test_mt_ag_seqs = [mt_ag_seqs[i] for i in test_idxes]
        test_labels = [labels[i] for i in test_idxes]

        train_dataset = SeqDataset(
            wt_ab_seqs=train_wt_ab_seqs,
            mt_ab_seqs=train_mt_ab_seqs,
            wt_ag_seqs=train_wt_ag_seqs,
            mt_ag_seqs=train_mt_ag_seqs,
            labels=train_labels,
        )

        test_dataset = SeqDataset(
            wt_ab_seqs=test_wt_ab_seqs,
            mt_ab_seqs=test_mt_ab_seqs,
            wt_ag_seqs=test_wt_ag_seqs,
            mt_ag_seqs=test_mt_ag_seqs,
            labels=test_labels,
        )
        return train_dataset, test_dataset

    def collator_fn(self, batch):
        wt_ab_seqs, mt_ab_seqs, wt_ag_seqs, mt_ag_seqs, labels = zip(*batch)
        wt_ab_inputs = self.tokenizer(
            wt_ab_seqs,
            padding=True,
            return_tensors="pt",
        )
        mt_ab_inputs = self.tokenizer(
            mt_ab_seqs,
            padding=True,
            return_tensors="pt",
        )
        wt_ag_inputs = self.tokenizer(
            wt_ag_seqs,
            padding=True,
            return_tensors="pt",
        )
        mt_ag_inputs = self.tokenizer(
            mt_ag_seqs,
            padding=True,
            return_tensors="pt",
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

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
        )

    def get_test_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
        )
