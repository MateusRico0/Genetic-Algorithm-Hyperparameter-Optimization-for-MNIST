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

## Getting Started

Follow these steps to set up your environment and run the project.

### 1. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python3.11 -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Run the Experiment

You can run the main experiment using the `genetic_network.py` script. You must specify which GA variant you want to run.
```bash
python3.11 genetic_network.py <variant_name>
```
Available Variants:

- standard
- lhs
- sus
- uniform
- adaptive
- steady-state

**Example:** To run the experiment with the "standard" genetic algorithm, use:
```bash
python3.11 genetic_network.py standard
```


## Project Files

- `genetic_network_parallel.py`
   - The main, parallelized Python script for running experiments. This will run the entire process from start to finish and generate the data.
- `genetic_networkl.py`
   - The same script but without multiprocessing
- `Genetic Network Selection.ipynb`
   - A Jupyter Notebook that allows you to run the same algorithm step-by-step. This is ideal for analysis, visualization, and debugging.

## Acknowledgements

Special thanks to Daniel Gardin. He first helped get the experiment running on the cluster. Then, arriving early the next day, he noticed a performance bottleneck and optimized the code with multiprocessing.
