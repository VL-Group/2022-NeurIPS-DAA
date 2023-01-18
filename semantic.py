"""
# Pytorch implementation for NeurIPS2022 paper from
# https://openreview.net/pdf?id=-KPNRZ8i0ag.
# "A Differentiable Semantic Metric Approximation in Probabilistic Embedding for Cross-Modal Retrieval"
# Hao Li, Jingkuan Song, Lianli Gao, Pengpeng Zeng, Haonan Zhang, Gongfu Li
#
# Writen by Hao Li, 2022
"""

import os
import string
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


def cider_compute(tfidf, eps=1e-6):
    """
    Shape
    -----
    Input1 : (torch.Tensor, torch.Tensor)
        :math:`((N, K, D), (N, D))` shape, `N` is the batch size, `K` is the number of the ground-truth captions and `D` is the length of the vocabulary.
    Output: torch.Tensor
        :math:`(N, N)`. The semantic score matrix computed by CIDEr.
    """
    (tfidf_GT, tfidf_single) = tfidf
    N = len(tfidf_single)
    tfidf_GT = tfidf_GT.view(-1, tfidf_GT.shape[-1]).to('cuda')
    tfidf_single = tfidf_single.to('cuda')

    cider_map_ = torch.mm(tfidf_GT, tfidf_single.t())

    cider_map = cider_map_.view(N , 5, -1)
    cider_map = cider_map.mean(1).squeeze(1)  #i2t: GT 2 single
    
    # cls_weight = cider_map * torch.eye(N).to('cuda')
    # cider_map = cider_map + cls_weight
    #cider_map = cider_map.view(-1, 5, N).mean(1).squeeze(1)  #[5000, 25000]

    return cider_map


def cider_compute_eval(tfidf, split='dev', eps=1e-6):
    """
    Shape
    -----
    Input1 : (torch.Tensor, torch.Tensor)
        :math:`((N, K, D), (N, D))` shape, `N` is the batch size, `K` is the number of the ground-truth captions and `D` is the length of the vocabulary.
    Output: torch.Tensor
        :math:`(N, N)`. The semantic score matrix computed by CIDEr.
    """
    (tfidf_GT, tfidf_single) = tfidf
    N = len(tfidf_single)
    tfidf_GT = tfidf_GT.view(-1, tfidf_GT.shape[-1]).to('cuda')
    tfidf_single = tfidf_single.to('cuda')

    cider_map_ = torch.mm(tfidf_GT, tfidf_single.t())

    cider_map = cider_map_.view(N , 5, -1)
    cider_map = cider_map.mean(1).squeeze(1)  #i2t: GT 2 single
    
    # cls_weight = cider_map * torch.eye(N).to('cuda')
    # cider_map = cider_map + cls_weight

    return cider_map

