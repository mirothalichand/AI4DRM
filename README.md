# AI4DRM: Artificial Intelligence for Disaster Risk Management

## ABOUT THE WORKSHOP

Welcome to an immersive deep learning experience!

This 3-day intensive workshop is designed to equip participants with comprehensive knowledge of neural networks, advanced architectures, and real-world applications in natural disaster prediction and detection systems.

Whether you're a student, researcher, or professional looking to advance your AI skills, this workshop combines theoretical foundations with hands-on practical sessions and expert insights from top academicians.

## Study Material

All of the study material will be available here, including the workshop notebooks and datasets.

This folder contains the workshop materials for the AI4DRM session on NeuralHydrology.

## Contents

- `workshop_material.ipynb`: A Jupyter Notebook designed to run in Google Colab. It guides you through setting up NeuralHydrology, preparing data, and training a Rainfall-Runoff model.

## How to Run

1.  **Google Colab**:
    - Download `workshop_material.ipynb`.
    - Go to [Google Colab](https://colab.research.google.com/).
    - Click "Upload" and select the notebook file.
    - Follow the instructions in the notebook.

2.  **Local Execution**:
    - Ensure you have Python installed.
    - Install Jupyter: `pip install jupyterlab`
    - Run `jupyter lab` and open the notebook.
    - *Note*: The notebook assumes it is running in an environment where it can clone the `neuralhydrology` repo. If you already have it, you may need to adjust the setup cells.

## Dataset

The workshop uses a small sample dataset included in the `neuralhydrology` repository (`test/test_data/camels_us`). This allows for quick training and evaluation without downloading the full 15GB CAMELS dataset.
