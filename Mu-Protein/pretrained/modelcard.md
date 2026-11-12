# Model Card for PMLM (Pairwise Masked Language Model)

PMLM is a protein sequence encoder pretrained using pairwise masked language modeling on UniRef50. It serves as the foundation model for downstream protein engineering tasks, including Mu-Protein, which fine-tunes PMLM for fitness prediction and search. This model captures contextual representations of amino acids that enable robust transfer to low-data protein tasks.

## Model Details

### Model Description

PMLM is trained with a BERT-style masked language modeling (MLM) objective with pairwise loss. It learns to predict randomly masked amino acids from their context in protein sequences. The model is implemented in PyTorch and trained on Microsoft Azure GPU infrastructure.

* **Developed by:** Microsoft Research AI for Science
* **Funded by \[optional]:** Microsoft
* **Shared by \[optional]:** Microsoft Research
* **Model type:** Protein sequence encoder
* **Language(s):** Not applicable (bio sequence model)
* **License:** MIT
* **Finetuned from model:** Not applicable

### Model Sources

* **Repository:** [https://github.com/microsoft/mu-protein](https://github.com/microsoft/mu-protein)
* **Paper:** [https://www.biorxiv.org/content/10.1101/2023.11.16.565910v5](https://www.biorxiv.org/content/10.1101/2023.11.16.565910v5)

## Uses

### Direct Use

Used as a pretrained encoder for extracting protein sequence embeddings, for tasks such as mutation effect prediction, protein design, or transfer learning to classification tasks.

### Downstream Use

PMLM is used as the backbone encoder in the Mu-Protein framework for protein fitness prediction and optimization. It can also be fine-tuned for other structure-free protein function prediction tasks.

### Out-of-Scope Use

* PMLM is not designed for protein structure prediction.
* Not suitable for generating novel sequences without a search or sampling mechanism.

## Bias, Risks, and Limitations

The model is trained on UniRef50, which may not represent all protein families equally. It may underperform on underrepresented domains or synthetic/engineered proteins that differ from natural proteins.

### Recommendations

Users should validate PMLM-based models on domain-specific data. For low-data scenarios, careful finetuning and domain adaptation is recommended.

## How to Get Started with the Model

Please refer to the repository for usage example.

## Training Details

### Training Data

* **Dataset:** UniRef50
* **Filtering:** Sequences deduplicated at 50% identity, cleaned for non-canonical amino acids.

### Training Procedure

#### Preprocessing

* Tokenization using a fixed 21-token amino acid vocabulary

#### Training Hyperparameters

* **Batch size:** dynamic
* **Optimizer:** AdamW
* **Precision:** mixed precision (fp16)
* **Learning rate:** 1e-5

#### Speeds, Sizes, Times

* **Model size:** \~650M parameters
* **Training time:** \~100 days on 48 V100 GPUs
* **Compute:** \~7 million PFLOPs

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

* Protein benchmark tasks
* Zero-shot fitness prediction on mutation effect datasets

#### Factors

* Sequence length

#### Metrics

* Perplexity
* Downstream transfer performance

### Results

PMLM enables accurate finetuning for downstream protein engineering tasks.

#### Summary

PMLM embeddings significantly outperform baseline models and support SOTA downstream performance in Mu-Protein.

## Environmental Impact

* **Hardware Type:** NVIDIA V100 GPUs
* **Hours used:** \~100,000 GPU hours
* **Cloud Provider:** Microsoft Azure
* **Compute Region:** East US
* **Carbon Emitted:** Estimated \~1,100 kg CO2e (based on mlco2 calculator)

## Technical Specifications

### Model Architecture and Objective

Transformer encoder with MLM objective, inspired by BERT. Sequence length up to 512.

### Compute Infrastructure

#### Hardware

48 x NVIDIA V100 (32GB)

#### Software

PyTorch, Fairseq

## Citation

**BibTeX:**

```bibtex
@article{he2023muprotein,
  title={Accelerating Protein Engineering with Fitness Landscape Modeling and Reinforcement Learning},
  author={Sun, Haoran and He, Liang and Deng, Pan and Liu, Guoqing and others},
  journal={bioRxiv},
  year={2023},
}
```

## Model Card Contact

Liang He ([liang.he@microsoft.com](mailto:liang.he@microsoft.com)),
Pan Deng ([pan.deng@microsoft.com](mailto:pan.deng@microsoft.com)),
Guoqing Liu ([guoqingliu@microsoft.com](mailto:guoqingliu@microsoft.com))

