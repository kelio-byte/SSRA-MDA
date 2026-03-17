# SSRA-MDA
PyTorch implementation for "SSRA-MDA: Structural–Semantic Representation Alignment for Microbe–Disease Association Prediction"

## Overview

In this paper, we propose **SSRA-MDA** (Structural–Semantic Representation Alignment), a deep learning framework for microbe–disease association prediction. The framework integrates structural information from association networks with semantic representations derived from biological ontologies using large language models (LLMs), and aligns the two views to learn more informative embeddings.

## Framework

<div align="center">
  <img src="framework.png" width="80%" alt="Framework">
</div>

## Usage

The main experiments are implemented in the Jupyter Notebook **mda.ipynb**.

To reproduce the experiments:

1. Clone the repository and install the required packages

```bash
git clone https://github.com/yourusername/SSRA-MDA.git
cd SSRA-MDA

pip install -r requirements.txt
```
2. Open and run mda.ipynb

## Project Structure

```
SSRA-MDA
│
├── HMDAD/               # HMDAD dataset
├── Disbiome/            # Disbiome dataset
├── case-study/          # Case study experiments
│
├── mda.ipynb            # Main notebook for training and evaluation
├── framework.png        # Framework illustration of SSRA-MDA
├── requirements.txt     # Python dependencies
└── README.md
```
