# Radial Basis Function Network with K-Means

This project implements a **Radial Basis Function (RBF) Neural Network** for classification tasks, using **K-Means clustering** to determine hidden layer prototypes. It also includes a Jupyter Notebook (`rbf.ipynb`) for experimentation and visualization.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

---

## Overview
- **Goal**: Explore RBF neural networks for supervised classification.
- **Key Features**:
  - Custom **K-Means implementation** (`kmeans.py`) for unsupervised clustering.
  - RBF network (`rbf_net.py`) with hidden layer activations based on cluster prototypes.
  - Linear regression-based weight training between hidden and output layers.
  - Functions for prediction, accuracy scoring, and confusion matrix generation.
  - Visualization utilities for predictions and clustering.

---

## Project Structure
.
├── rbf.ipynb # Jupyter Notebook for running experiments and visualizations
├── rbf_net.py # Implementation of the RBF Neural Network
├── kmeans.py # Implementation of K-Means clustering
└── README.md # Project documentation



---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd <repo-folder>

---
## Acknowledgements
Course: CS 251 – Data Analysis and Visualization


