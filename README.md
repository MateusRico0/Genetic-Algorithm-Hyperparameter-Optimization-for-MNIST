# Genetic Algorithm Hyperparameter Optimization for MNIST

## Overview

This project implements **six types of Genetic Algorithms (GAs)** to search for the best hyperparameters of a simple Convolutional Neural Network (CNN) trained on the **MNIST dataset**.

Each GA variant explores a different selection, crossover, or mutation strategy, and the results are compared in terms of validation accuracy and convergence speed.

The script builds, trains, and evaluates small CNN models using _TensorFlow/Keras_, and visualizes performance comparisons between GA variants.

---

## Features

### Genetic Algorithm Variants
1. **Standard GA** – Basic implementation with roulette wheel selection, single-point crossover, and fixed mutation rate.  
2. **Latin Hypercube Sampling (LHS)** – Uses LHS for population initialization to ensure better coverage of the search space.  
3. **Stochastic Universal Sampling (SUS)** – Selection technique ensuring proportional sampling with lower variance.  
4. **Uniform Crossover GA** – Performs crossover per gene with a uniform swap probability instead of a fixed crossover point.  
5. **Adaptive Mutation GA** – Mutation rate dynamically adjusts depending on the stagnation of the best fitness value.  
6. **Steady-State GA** – Replaces only a few of the worst individuals per generation instead of full generational replacement.

---

## Hyperparameters Being Optimized

The Genetic Algorithm searches for the best combination of the following hyperparameters:

| Hyperparameter       | Type           | Range/Values                       |
|----------------------|----------------|------------------------------------|
| Learning rate        | Continuous     | 1e-4 to 1e-1 (log scale)          |
| Kernel size          | Categorical    | {3, 5, 7}                         |
| Batch size           | Categorical    | {32, 64, 128}                     |
| Optimizer            | Categorical    | {sgd, adam, rmsprop}              |
| Number of hidden layers | Integer     | {1, 2, 3}                         |
| Neurons per layer    | Categorical    | {32, 64, 128, 256}                |

Each individual in the population encodes a complete set of these hyperparameters.

---

## How It Works

1. **Initialization**  
   A population of candidate solutions (hyperparameter sets) is generated. Depending on the GA variant, this may use random sampling or Latin Hypercube Sampling.

2. **Evaluation**  
   Each individual’s fitness is evaluated by training a small CNN on the MNIST dataset and recording its validation accuracy.

3. **Selection**  
   Parents are chosen for reproduction based on their fitness (e.g., roulette wheel or SUS selection).

4. **Crossover & Mutation**  
   Genetic operators create new offspring by combining and slightly modifying parent hyperparameters.

5. **Replacement**  
   Depending on the GA type, new individuals either replace the entire population or only the worst performers.

6. **Termination**  
   The process continues for a specified number of generations. The best hyperparameters and validation accuracy are recorded.

---

## Requirements

- Python 3.11
- TensorFlow / Keras
- NumPy
- Matplotlib


## How to Run

**Create a virtual environment**
```bash
python3.11 -m venv venv
```

**Install dependencies using:**
```bash
pip install -r requirements.txt
```
**Run all cells from the Genetic Network Selection Notebook:**


