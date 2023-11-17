# MultiEMO: An Attention-Based Correlation-Aware Multimodal Fusion Framework for Emotion Recognition in Conversations (ACL 2023)

## Overview
This repository is the Pytorch implementation of ACL 2023 paper ["MultiEMO: An Attention-Based Correlation-Aware Multimodal Fusion Framework for Emotion Recognition in Conversations"](https://aclanthology.org/2023.acl-long.824.pdf). In this work, we propose a novel attention-based correlation-aware multimodal fusion framework named MultiEMO,
which effectively integrates multimodal cues by capturing cross-modal mapping relationships across textual, audio and visual modalities based on bidirectional multi-head crossattention layers.

## Quick Start
### Clone the repository
```
git clone https://github.com/TaoShi1998/MultiEMO-ACL2023.git
```
### Environment setup
```
# Environment: Python 3.6.8 + Torch 1.10.0 + CUDA 11.3
# Hardware: single RTX 3090 GPU, 256GB RAM
conda create --name MultiEMOEnv python=3.6
conda activate MultiEMOEnv
```
### Install dependencies
```
cd MultiEMO
pip install -r requirements.txt
```
### Run the model
```
# IEMOCAP Dataset
bash Train/TrainMultiEMO_IEMOCAP.sh

# MELD Dataset
bash Train/TrainMultiEMO_MELD.sh
```

## Citation
If you find our work helpful to your research, please cite our paper as follows.
```bibtex
@inproceedings{shi-huang-2023-multiemo,
    title = "{M}ulti{EMO}: An Attention-Based Correlation-Aware Multimodal Fusion Framework for Emotion Recognition in Conversations",
    author = "Shi, Tao  and
      Huang, Shao-Lun",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.824",
    doi = "10.18653/v1/2023.acl-long.824",
    pages = "14752--14766",
    abstract = "Emotion Recognition in Conversations (ERC) is an increasingly popular task in the Natural Language Processing community, which seeks to achieve accurate emotion classifications of utterances expressed by speakers during a conversation. Most existing approaches focus on modeling speaker and contextual information based on the textual modality, while the complementarity of multimodal information has not been well leveraged, few current methods have sufficiently captured the complex correlations and mapping relationships across different modalities. Furthermore, existing state-of-the-art ERC models have difficulty classifying minority and semantically similar emotion categories. To address these challenges, we propose a novel attention-based correlation-aware multimodal fusion framework named MultiEMO, which effectively integrates multimodal cues by capturing cross-modal mapping relationships across textual, audio and visual modalities based on bidirectional multi-head cross-attention layers. The difficulty of recognizing minority and semantically hard-to-distinguish emotion classes is alleviated by our proposed Sample-Weighted Focal Contrastive (SWFC) loss. Extensive experiments on two benchmark ERC datasets demonstrate that our MultiEMO framework consistently outperforms existing state-of-the-art approaches in all emotion categories on both datasets, the improvements in minority and semantically similar emotions are especially significant.",
}
```
