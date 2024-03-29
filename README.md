![Python](https://img.shields.io/badge/Python-v3.7.9-orange.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.7.0-blue.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-v10.1-green.svg?style=plastic)
![transformers](https://img.shields.io/badge/transformers-v4.10.0-pink.svg?style=plastic)

# DiSCoMaT: Distantly Supervised Composition Extraction from Tables in Materials Science

## About
This repository contains the official implementation of 
[DiSCoMaT: Distantly Supervised Composition Extraction from Tables in Materials Science](https://comingsoon). This model is trained on **4,408 distantly supervised tables** published in materials science research papers to extract compositions reported in the tables. These tables yielded a total of 38,799 tuples in the training set.

The tuples are of the form $\{(id, c_k^{id}, p_k^{id}, u_k^{id} )\}_{k=1}^{K^{id}}$, where
- $id$ is the id of the material composition reported in the tables
- $c_k^{id}$ is the k-th chemical element in the material composition
- $p_k^{id}$ is the percentage of k-th chemical element in the material composition
- $u_k^{id}$ is the unit of the of $p_k^{id}$ (either mole % or weight %)


The following figure represents the architecture of the model proposed in our work.

![DiSCoMaT](data/discomat.png)

## Notes
- The [**code**](code) directory contains the file for training models reported in this paper.
- The [**data**](data) directory contains the dataset for training models reported in this paper.
- The [**notebooks**](notebooks) directory contains Jupyter notebook to visualise the dataset.
- The respective directories and sub-directories contain task-specific **README** files.

## Citation
If you find this repository useful, please cite our work as follows:
```
@inproceedings{gupta-etal-2023-discomat,
    title = "{D}i{SC}o{M}a{T}: Distantly Supervised Composition Extraction from Tables in Materials Science Articles",
    author = "Gupta, Tanishq  and
      Zaki, Mohd  and
      Khatsuriya, Devanshi  and
      Hira, Kausik  and
      Krishnan, N M Anoop  and
      -, Mausam",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.753",
    pages = "13465--13483",
}
```
