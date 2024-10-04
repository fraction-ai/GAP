markdown_content = """
# GAMIFIED CROWD-SOURCING OF HIGH-QUALITY DATA FOR VISUAL FINE-TUNING

This repository contains the code and dataset for the paper: **Gamified Adversarial Prompting (GAP)** - *A Framework for Crowd-Sourcing High-Quality Data for Visual Fine-Tuning*. This work proposes a unique approach to collecting high-quality data for large multimodal models through a gamified, crowd-sourcing methodology.

## Overview

The Gamified Adversarial Prompting (GAP) framework aims to identify weaknesses in large multimodal models (LMMs) by crowd-sourcing adversarial visual questions. It transforms data collection into an engaging game, encouraging human participants to challenge the AI model with questions it may not answer correctly. This process gathers valuable question-answer pairs that enhance visual instruction tuning.

### Key Contributions:
1. **Gamified Data Collection**: We introduce a game where participants earn points by asking challenging questions that lead to model errors.
2. **Scalable Framework**: The system scales up rapidly, with over 50,000 participants contributing to the dataset.
3. **Model Fine-Tuning**: The GAP framework improves the accuracy and robustness of LMMs by addressing gaps in their knowledge.

## Dataset and Gamified Framework

The GAP framework uses two datasets derived from MS-COCO:
- **Tainted dataset**: Contains images that are specifically selected to be simple and well understood by the model.
- **Untainted dataset**: Consists of more complex images, used to challenge the model’s understanding.

Players are presented with images and tasked with finding a question that the AI model cannot answer correctly. The system logs interactions, and players earn rewards based on the correctness of their adversarial questions. The data generated from this process is then used to fine-tune large multimodal models.

### Gameplay:
1. Player is presented with an image.
2. The player asks a question that the AI is expected to answer incorrectly.
3. The player marks whether the AI answered correctly or not.
4. Points are awarded based on the difficulty and accuracy of the interaction.

---

## Model Performance

The GAP-generated dataset was used to fine-tune several multimodal models, resulting in significant improvements in their visual question answering (VQA) capabilities. The models evaluated include:
- **MiniCPM-Llama3-V-2.5-8B**
- **Qwen2-VL-2B**
- **Qwen2-VL-7B**

### Performance Improvements

| Model                   | Pre-Fine-tuning GPT Score | Post-Fine-tuning GPT Score | Improvement |
|-------------------------|---------------------------|----------------------------|-------------|
| GPT-4V (Benchmark)       | 0.637                     | -                          | -           |
| MiniCPM-Llama3-V-2.5-8B | 0.147                     | **0.477**                  | +0.300      |
| Qwen2-VL-2B             | 0.169                     | **0.285**                  | +0.116      |
| Qwen2-VL-7B             | 0.207                     | **0.250**                  | +0.043      |

The **MiniCPM-Llama3-V-2.5-8B** model demonstrated significant improvement (+0.300) after being fine-tuned using the GAP-generated dataset, nearly closing the gap with GPT-4V in some respects. The smaller Qwen2 models also exhibited notable gains.

### Benchmark Performance (MiniCPM-Llama3-V-2.5-8B)

| Benchmark       | Pre-Fine-tuning | Post-Fine-tuning |
|-----------------|-----------------|------------------|
| LLaVA Bench     | **87.9**        | 82.2             |
| OCRBench        | 72.4            | **73.1**         |
| MME             | 2025.61         | **2040.54**      |
| RealWorldQA     | **0.634**       | 0.609            |
| MM-Vet          | 51.422          | **51.789**       |
| MMBench         | **0.752**       | 0.7422           |
| HallusionBench  | 59.93           | **60.25**        |
| TextVQA         | 76.63           | **76.966**       |
| MMMU val        | **0.474**       | 0.486            |
| DocVQA          | **84.47**       | 84.33            |

For **MiniCPM-Llama3-V-2.5-8B**, we observed improvements in several critical benchmarks, including **OCRBench**, **MM-Vet**, and **HallusionBench**, which focus on complex visual-textual reasoning, reading text from images, and reducing hallucinations, respectively. There were some slight regressions in **RealWorldQA** and **MMMU**, but overall performance was improved post-fine-tuning.

---

## Cross-Model Evaluation

One of the key insights of the GAP framework is that the fine-tuning data is not only useful for the model it was collected from but also benefits other models. We evaluated **Qwen2-VL-2B** and **Qwen2-VL-7B** on the GAP dataset and observed significant cross-model performance improvements.

### Cross-Model Evaluation (Qwen2-VL-7B)

| Benchmark        | Pre-Fine-tuning | Post-Fine-tuning |
|------------------|-----------------|------------------|
| LLaVA Bench      | 76.7            | **83.6**         |
| OCRBench         | 86.1            | **86.7**         |
| MME              | 2318.98         | **2332.71**      |
| RealWorldQA      | **0.699**       | 0.690            |
| MM-Vet           | 62.889          | **64.954**       |
| MMBench          | 0.808           | **0.815**        |
| HallusionBench   | 68.769          | 68.769           |
| TextVQA          | **84.428**      | 84.084           |
| MMMU val         | 0.524           | **0.527**        |
| DocVQA           | 93.866          | **94.038**       |

**Qwen2-VL-7B** demonstrated improvements in benchmarks like **MM-Vet**, **OCRBench**, and **MMBench**. This suggests that fine-tuning on the GAP dataset helps the model generalize across different tasks involving complex visual and textual reasoning.

### Cross-Model Evaluation (Qwen2-VL-2B)

| Benchmark        | Pre-Fine-tuning | Post-Fine-tuning |
|------------------|-----------------|------------------|
| LLaVA Bench      | 52.6            | **57.9**         |
| OCRBench         | 81.2            | **81.4**         |
| MME              | 1881.92         | **1962.75**      |
| RealWorldQA      | **0.626**       | 0.6156           |
| MM-Vet           | 51.146          | **52.43**        |
| MMBench          | 0.729           | **0.732**        |
| HallusionBench   | 61.619          | **62.99**        |
| TextVQA          | **79.824**      | 80.074           |
| MMMU val         | 0.414           | **0.448**        |
| DocVQA           | **89.26**       | 89.36            |

**Qwen2-VL-2B** showed notable gains in **LLaVA Bench**, **MM-Vet**, and **HallusionBench**. The model benefits significantly from GAP fine-tuning in areas such as text reading and hallucination detection.

---

## Experimental Setup

The GAP-VQA dataset consists of over 3,600 question-image pairs, designed to test various visual understanding tasks such as object recognition, spatial reasoning, counting, and text reading. These data points were split into **GAP-VQA-train** for fine-tuning and **GAP-VQA-val** for validation.

Fine-tuning was conducted using **LoRA** (Low-Rank Adaptation) to minimize GPU usage while maximizing model performance improvements. This allowed for faster experimentation and reduced computational costs.

---

## Reward System

To encourage engagement and high-quality data submissions, the GAP framework incorporates a points-based reward system:
- Players earn **20 points** for each valid adversarial question where the model fails.
- **Leaderboards** and **Web3 airdrops** incentivize consistent participation.
- **Cash prizes** are awarded to top players weekly.

The combination of intrinsic motivation (sense of accomplishment) and extrinsic rewards (points, prizes) ensures sustained participation and high-quality data collection.

---

## Conclusion

The **Gamified Adversarial Prompting (GAP)** framework introduces a novel approach to improving large multimodal models by crowd-sourcing high-quality data through gamification. The results show substantial improvements in model performance, with clear cross-model benefits. GAP not only enhances visual question
