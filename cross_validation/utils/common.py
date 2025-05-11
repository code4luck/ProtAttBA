"""
读取csv文件获取所有需要的条目
https://www.cnblogs.com/jhcelue/p/7248465.html#:~:text=%E5%9C%A8%E5%9F%BA%E4%BA%8E%E8%9B%8B%E7%99%BD%E8%B4%A8%E5%BA%8F%E5%88%97%E7%9A%84%E7%9B%B8
https://blog.csdn.net/cbb_ft/article/details/124623766
"""

import pandas as pd
import numpy as np
import os, shutil
from collections import defaultdict


def read_s645_csv(file_path):
    csv = pd.read_csv(
        file_path,
        usecols=[
            "PDB",
            "Mutation",
            "antibody_light_seq",
            "antibody_heavy_seq",
            "antigen_a_seq",
            "antigen_b_seq",
            "antibody_light_seq_mut",
            "antibody_heavy_seq_mut",
            "antigen_a_seq_mut",
            "antigen_b_seq_mut",
            "ddG",
        ],
    )
    names = csv["PDB"].tolist()
    abls = csv["antibody_light_seq"].tolist()
    abhs = csv["antibody_heavy_seq"].tolist()
    agas = csv["antigen_a_seq"].tolist()
    agbs = csv["antigen_b_seq"].tolist()
    abls_m = csv["antibody_light_seq_mut"].tolist()
    abhs_m = csv["antibody_heavy_seq_mut"].tolist()
    agas_m = csv["antigen_a_seq_mut"].tolist()
    agbs_m = csv["antigen_b_seq_mut"].tolist()
    labels = csv["ddG"].tolist()
    return names, abls, abhs, agas, agbs, abls_m, abhs_m, agas_m, agbs_m, labels


def cat_seq(seq1: str, seq2: str):
    if isinstance(seq1, float):
        return seq2
    if isinstance(seq2, float):
        return seq1
    return seq1 + seq2


def get_ab645_pair_data(file_path):
    names, abls, abhs, agas, agbs, abls_m, abhs_m, agas_m, agbs_m, labels = (
        read_s645_csv(file_path)
    )
    assert (
        len(abls)
        == len(abhs)
        == len(agas)
        == len(agbs)
        == len(abls_m)
        == len(abhs_m)
        == len(agas_m)
        == len(agbs_m)
        == len(labels)
    ), "data length not consistent"
    ab_hl_chains = []
    ag_ab_chains = []
    mt_ab_hl_chains = []
    mt_ag_ab_chains = []
    for i in range(len(names)):
        ab_hl_chains.append(cat_seq(abls[i], abhs[i]))
        ag_ab_chains.append(cat_seq(agas[i], agbs[i]))
        mt_ab_hl_chains.append(cat_seq(abls_m[i], abhs_m[i]))
        mt_ag_ab_chains.append(cat_seq(agas_m[i], agbs_m[i]))
    return ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels


def read_AB1101_csv(file_path):
    csv = pd.read_csv(
        file_path,
        usecols=[
            "PDB",
            "Mutation",
            "antibody_light_seq",
            "antibody_heavy_seq",
            "antigen_a_seq",
            "antigen_b_seq",
            "antibody_light_seq_mut",
            "antibody_heavy_seq_mut",
            "antigen_a_seq_mut",
            "antigen_b_seq_mut",
            "ddG",
        ],
    )
    names = csv["PDB"].tolist()
    abls = csv["antibody_light_seq"].tolist()
    abhs = csv["antibody_heavy_seq"].tolist()
    agas = csv["antigen_a_seq"].tolist()
    agbs = csv["antigen_b_seq"].tolist()
    abls_m = csv["antibody_light_seq_mut"].tolist()
    abhs_m = csv["antibody_heavy_seq_mut"].tolist()
    agas_m = csv["antigen_a_seq_mut"].tolist()
    agbs_m = csv["antigen_b_seq_mut"].tolist()
    labels = csv["ddG"].tolist()
    return names, abls, abhs, agas, agbs, abls_m, abhs_m, agas_m, agbs_m, labels


def read_s1131_csv(file_path):
    csv = pd.read_csv(
        file_path, usecols=["PDB", "mutation", "a", "b", "a_mut", "b_mut", "ddG"]
    )
    names = csv["PDB"].tolist()
    ab = csv["a"].tolist()
    ag = csv["b"].tolist()
    ab_m = csv["a_mut"].tolist()
    ag_m = csv["b_mut"].tolist()
    labels = csv["ddG"].tolist()
    return names, ab, ag, ab_m, ag_m, labels


def get_ab1101_pair_data(file_path):
    names, abls, abhs, agas, agbs, abls_m, abhs_m, agas_m, agbs_m, labels = (
        read_AB1101_csv(file_path)
    )
    assert (
        len(abls)
        == len(abhs)
        == len(agas)
        == len(agbs)
        == len(abls_m)
        == len(abhs_m)
        == len(agas_m)
        == len(agbs_m)
        == len(labels)
    ), "data length not consistent"
    ab_hl_chains = []
    ag_ab_chains = []
    mt_ab_hl_chains = []
    mt_ag_ab_chains = []
    for i in range(len(names)):
        ab_hl_chains.append(cat_seq(abls[i], abhs[i]))
        ag_ab_chains.append(cat_seq(agas[i], agbs[i]))
        mt_ab_hl_chains.append(cat_seq(abls_m[i], abhs_m[i]))
        mt_ag_ab_chains.append(cat_seq(agas_m[i], agbs_m[i]))
    return ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels


def get_s1131_data(file_path):
    names, ab, ag, ab_m, ag_m, labels = read_s1131_csv(file_path)
    labels = np.array(labels)  # [B,]
    return ab, ag, ab_m, ag_m, labels

