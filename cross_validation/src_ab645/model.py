
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import EsmModel

from model_module.rope_attn import MutilHeadSelfAttn


class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class AttnMean(nn.Module):
    """
    采用conv_attn进行计算权重并进行加权平均
    """

    def __init__(
        self,
        hidden_size,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        x = self.layer_norm(x)
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)  # [N, L]
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        # attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)  # [B, L, 1]
        # x = (attn * x).sum(dim=1)
        attn = F.softmax(attn, dim=-1)  # [B, L]
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(
            1
        )  # [B, 1, L] * [B, L, H] -> [B, 1, H] --> [B, H]
        return x


class AttnTransform(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        x = self.layer_norm(x)
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)  # [N, L]
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)  # [B, L, 1]
        x = attn * x  # [B, L, H]
        return x


class OutHead(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ac = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.re = nn.ReLU()
        self.linear = nn.Linear(hidden_dim // 2, out_dim)

    def forward(self, x):
        x = self.ac(self.fc1(x))
        x = self.dropout(x)
        x = self.re(self.fc2(x))
        x = self.linear(x)
        return x


class SeqBindModel(nn.Module):
    def __init__(self, args):
        super(SeqBindModel, self).__init__()
        self.config = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        self.encoder = EsmModel.from_pretrained(args.model_locate)
        self.wt_ab_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.wt_ag_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.mt_ab_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.mt_ag_conv_transformer = AttnTransform(hidden_size=self.hidden_size)

        self.wt_ab_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads, hidden_dim=self.hidden_size, dropout=self.dropout
        )
        self.mt_ab_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads, hidden_dim=self.hidden_size, dropout=self.dropout
        )
        self.wt_ag_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads, hidden_dim=self.hidden_size, dropout=self.dropout
        )
        self.mt_ag_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads, hidden_dim=self.hidden_size, dropout=self.dropout
        )

        self.wt_ab_mean = AttnMean(self.hidden_size)
        self.mt_ab_mean = AttnMean(self.hidden_size)
        self.wt_ag_mean = AttnMean(self.hidden_size)
        self.mt_ag_mean = AttnMean(self.hidden_size)

        self.bn = nn.BatchNorm1d(self.hidden_size * 2)

        self.out_head = OutHead(self.hidden_size * 2, 1)

        if self.config.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        wt_ab_inputs_ids,  
        wt_ab_inputs_mask,  
        mut_ab_inputs_ids, 
        mt_ab_inputs_mask, 
        wt_ag_inputs_ids, 
        wt_ag_inputs_mask,  
        mut_ag_inputs_ids,  
        mt_ag_inputs_mask,  
        wt_features=None,
        mt_features=None,
        labels=None,
    ):
        wt_ab_embeddings = self.encoder(
            input_ids=wt_ab_inputs_ids, attention_mask=wt_ab_inputs_mask
        ).last_hidden_state  # [B, L, H]
        mt_ab_embeddings = self.encoder(
            input_ids=mut_ab_inputs_ids, attention_mask=mt_ab_inputs_mask
        ).last_hidden_state  # [B, L, H]

        wt_ag_embeddings = self.encoder(
            input_ids=wt_ag_inputs_ids, attention_mask=wt_ag_inputs_mask
        ).last_hidden_state
        mt_ag_embeddings = self.encoder(
            input_ids=mut_ag_inputs_ids, attention_mask=mt_ag_inputs_mask
        ).last_hidden_state


        wt_ab_embeddings = self.wt_ab_conv_transformer(
            wt_ab_embeddings, wt_ab_inputs_mask
        )
        mt_ab_embeddings = self.mt_ab_conv_transformer(
            mt_ab_embeddings, mt_ab_inputs_mask
        )

        wt_ag_embeddings = self.wt_ag_conv_transformer(
            wt_ag_embeddings, wt_ag_inputs_mask
        )
        mt_ag_embeddings = self.mt_ag_conv_transformer(
            mt_ag_embeddings, mt_ag_inputs_mask
        )


        wt_ag_embedding = self.wt_ag_attn(
            wt_ag_embeddings, wt_ab_embeddings, wt_ab_embeddings, mask=wt_ab_inputs_mask
        )  # [B, L_q, H]

        mt_ag_embedding = self.mt_ag_attn(
            mt_ag_embeddings, mt_ab_embeddings, mt_ab_embeddings, mask=mt_ab_inputs_mask
        )  # [B, L_q, H]

        wt_ab_embedding = self.wt_ab_attn(
            wt_ab_embeddings, wt_ag_embeddings, wt_ag_embeddings, mask=wt_ag_inputs_mask
        )
        mt_ab_embedding = self.mt_ab_attn(
            mt_ab_embeddings, mt_ag_embeddings, mt_ag_embeddings, mask=mt_ag_inputs_mask
        )


        wt_ag_embedding = self.wt_ag_mean(wt_ag_embedding, wt_ag_inputs_mask) 
        mt_ag_embedding = self.mt_ag_mean(mt_ag_embedding, mt_ag_inputs_mask) 
        wt_ab_embedding = self.wt_ab_mean(wt_ab_embedding, wt_ab_inputs_mask)
        mt_ab_embedding = self.mt_ab_mean(mt_ab_embedding, mt_ab_inputs_mask)
        wt_abag = wt_ag_embedding + wt_ab_embedding 
        mt_abag = mt_ag_embedding + mt_ab_embedding 

        rep = torch.cat([wt_abag, mt_abag], dim=1) 
        rep = self.bn(rep)
        out = self.out_head(rep)
        return out.squeeze(1)  # [B, 1] -> [B]

