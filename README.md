# GaMS3-12B Training code

This repository contains the official training code and configuration files for **GaMS3-12B-Instruct**, the next generation of the Generative Model for Slovene (GaMS).

Based on the **Gemma 3-12B** architecture, this model has been significantly enhanced through multi-stage continual pre-training (CPT) and supervised fine-tuning (SFT) to excel in Slovene, while maintaining high performance in English and other South Slavic languages (Croatian, Serbian, and Bosnian).

---

## 🚀 Overview

The development of GaMS3-12B-Instruct followed a rigorous five-stage training pipeline designed to adapt a global base model to a specific linguistic and cultural context without losing general reasoning capabilities.

### Key Features

* **Base Model:** `google/gemma-3-12b-pt`
* **Languages:** Primary focus on **Slovene** and **English**; secondary support for **Croatian, Serbian, and Bosnian**.
* **Context Window:** Trained with support for up to **131,072 tokens**.
* **Frameworks:** Built using **NVIDIA NeMo Framework 2.0** (for CPT) and **Hugging Face Transformers/DeepSpeed/TRL** (for SFT).

---

## 🛠️ Repository Structure

* `/pretraining`: Configuration and scripts for NVIDIA NeMo-based continual pre-training.
* `/sft`: Training scripts for Supervised Fine-Tuning using DeepSpeed and TRL.

---

## 🖥️ Code Usage & Infrastructure

This repository is designed for execution on high-performance computing (HPC) environments using the **SLURM** workload manager. Each training stage includes specialized SBATCH scripts tailored to the infrastructure used during development.

### 🏗️ Cluster Configurations

The scripts are organized based on the specific requirements of the training phase and the target hardware:

* **Parallel Alignment & Base CPT:**
  * Provided scripts are optimized for the **LEONARDO Booster partition** (EuroHPC).
  * Uses **Singularity** containers to manage the environment across a large-scale multi-node setup (up to 256 nodes).


* **Long CPT & SFT Stages:**
  * Scripts are optimized for **Pyxis/Enroot**-based clusters, specifically targeting a **single DGX B200 node**.
  * While configured for a single node, these scripts are designed to be easily extendable for multi-node scaling by adjusting the `#SBATCH --nodes` and distributed init parameters.



### 📦 Environment & Containers

To ensure reproducibility and performance, all stages rely on the official **NVIDIA NeMo Framework** container.

* **Base Container:** `nvcr.io/nvidia/nemo:25.09`
* **Library Management:** * For **CPT stages**, the NeMo container provides all necessary dependencies out-of-the-box.
* For **SFT stages**, the scripts are designed to be "plug-and-play." They will automatically detect and install any missing libraries (such as specific versions of `trl` or `peft`) on the fly within the container environment.



### 🚀 Running the Training

To launch a specific stage, navigate to the respective directory and submit the SBATCH script:

```bash
# Example: Launching the Base CPT stage on Leonardo
cd pretraining/base_cpt
sbatch run_pretraining.sbatch

# Example: Launching SFT on a DGX B200 node
cd sft/instruction_tuning
sbatch run_sft.sbatch

```

### Data Format

NeMo framework expects indexed memmap data for pretraining. The scripts for pretraining data preparation are available [here](https://github.com/GaMS-Team/GaMS3-12B-CPT-Data-Preparation).

For the SFT phases, the script expect JSONL datasets in the Prompt/completion format (see [GaMS-Nemotron-Chat](https://huggingface.co/datasets/cjvt/GaMS-Nemotron-Chat) dataset).

---

## 🏗️ Training Pipeline

The training was executed in five distinct stages to ensure linguistic alignment and safety.

### 1. Continual Pre-training (CPT)

Performed using **NVIDIA NeMo 2.0** on the Leonardo EuroHPC booster partition and NVIDIA DGX Cloud Lepton.

| Stage | Objective | Context Window |
| --- | --- | --- |
| **Parallel Alignment** | Alignment of EN and SL through parallel corpora. | 64k |
| **Base CPT** | Exposure to diverse corpora (SL, EN, HR, SR, BS). | 64k |
| **Long CPT** | High-quality data pre-training for long-context stability. | 128k |

### 2. Supervised Fine-tuning (SFT)

Performed using **Transformers** and **DeepSpeed ZeRO Stage 2**.

* **Base Instruction SFT:** 100k+ examples covering QA, writing, math, and code.
* **Chat & Safety Tuning:** Refinement for conversational flow and adherence to safety guidelines.

---

## 📊 Evaluation

GaMS3-12B-Instruct sets a new standard for open-source models in the 12B parameter range for the Slovene language.

### Slovenian LLM Eval Comparison (Average Rank)

| Model | Average Rank (Lower is better) |
| --- | --- |
| GaMS-27B-Nemotron | 3.05 |
| Gemma-3-27B-IT | 3.20 |
| **GaMS3-12B-Instruct** | **4.25** |
| Bielik-11B | 5.00 |
| Gemma-3-12B-IT | 6.30 |

---


## 🏛️ Acknowledgments

Developed by researchers at the **University of Ljubljana, Faculty of Computer and Information Science**, within the **PoVeJMo** research program.

The project was supported by:

* **ARIS** (Slovenian Research and Innovation Agency).
* **NextGenerationEU**.
* **NVIDIA Sovereign AI initiative**.
* **EuroHPC JU**.
* **SLING** (Slovenian National Supercomputing Network).
* **SLAIF** (Slovenian AI Factory).

### Team

* Domen Vreš, Iztok Lebar Bajec, Tjaša Arčon, Timotej Petrič, Dario Vajda, and Marko Robnik-Šikonja.

---

## ⚖️ License

This project is licensed under the **Apache-2.0 licensee**.
