# Convolutional Neural Networks - Exploring Layers & Inductive Bias

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![AWS](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazonaws)

> A deep dive into convolutional layers, architectural design choices, and their impact on learning and generalization.

---

## Overview

This repository focuses on **Convolutional Neural Networks (CNNs)** as a concrete example of how **inductive bias** is introduced into learning systems. Rather than treating neural networks as black boxes, this project explores how specific architectural choices—such as kernel size, stride, padding, and depth—directly affect model performance and interpretability.

> [!IMPORTANT]
> This project emphasizes **architectural reasoning** and **experimental rigor** over simple hyperparameter tuning.

| Aspect | Description |
| :--- | :--- |
| **Domain** | Computer Vision / Image Analysis |
| **Task** | Classification & Architectural Experimentation |
| **Method** | CNN (Convolutional Neural Networks) |
| **Infrastructure** | AWS SageMaker (Training & Deployment) |

---

## Project Structure

```text
/
├── README.md
├── notebooks/
│   └── lab03_cnn_experiments.ipynb
├── data/
│   └── lab03-moodle.md
├── src/
│   └── models/
└── img/
```


---

## Learning Objectives

By completing this assignment, we aim to:
1. Understand the mathematical intuition behind **convolutional layers**.
2. Analyze how **kernel size, depth, stride, and padding** affect the learning process.
3. Compare CNNs with **fully connected layers** for image datasets.
4. Perform meaningful **EDA** for computer vision tasks.
5. Train and deploy models using **AWS SageMaker**.

---

## Theoretical Background

### Convolutional Layer Inductive Bias
Convolutional layers introduce two main inductive biases:
1. **Locality**: Nearby pixels are more strongly related than distant ones.
2. **Translation Invariance**: A feature (like an edge) is relevant regardless of its position in the image.

---

## Author

**Sergio Andrey Silva Rodriguez**  
*Systems Engineering Student*  
Escuela Colombiana de Ingeniería Julio Garavito

---

<details>
<summary>License</summary>

This project is for educational purposes as part of the AREP course at Escuela Colombiana de Ingeniería Julio Garavito.

</details>
