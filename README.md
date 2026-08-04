# Handwritten Digit Classification Framework

A machine learning classification framework implemented entirely in **Java**
without relying on external machine learning libraries.

The project implements several commonly used supervised learning algorithms
from first principles to better understand their mathematical foundations,
training procedures and performance characteristics.

The framework provides implementations of:

- Nearest Neighbour (NN)
- k-Nearest Neighbour (k-NN)
- Weighted k-Nearest Neighbour
- Multilayer Perceptron (MLP)
- Support Vector Machine (SVM)

The primary goal of the project was to explore how these algorithms operate
internally rather than relying on existing machine learning frameworks.

---

## Features

- Pure Java implementation
- No external machine learning libraries
- Common interface for multiple classifiers
- Performance comparison across algorithms
- Training and evaluation on handwritten digit recognition

---

## Dataset

This project uses the **Optical Recognition of Handwritten Digits**
dataset from the UCI Machine Learning Repository.

Dataset:
https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

Creators:

- E. Alpaydin
- C. Kaynak

License:

Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset was created by the original authors and is used in accordance
with the dataset licence.

---

## Project Structure

```
src/
    classifiers/
        NN/
        KNN/
        WeightedKNN/
        MLP/
        SVM/

    dataset/

    evaluation/
```

---

## Documentation

Implementation details and algorithm descriptions are documented throughout
the source code.
