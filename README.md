# README

This project contains code for multiple datasets, with each dataset having its own folder. Below are the instructions for running the code for each dataset, as well as details about the Shapley value implementation.

## Running Instructions

Each dataset's code is located in its respective folder. You can navigate to the corresponding dataset folder and run the code as follows:

### 1. ADM Dataset

- Navigate to the `AD` folder:

  ```bash
  cd AD
  ```
- Run the following command:
  ```bash
  python MMFL_main.py
  ```

### 2. CMHAD Dataset
- Navigate to the `CMHAD` folder:
  ```bash
  cd CMHAD
  ```
- Run the following command:
  ```bash
  python MMFL_main.py
  ```

### 3. Flash Dataset
- Navigate to the `Flash` folder:
  ```bash
  cd Flash
  ```
- Run the following command:
  ```bash
  python MM_flash.py
  ```

### 4. MHAD Dataset
- Navigate to the `MHAD` folder:
  ```bash
  cd MHAD
  ```
- Run the following command:
  ```bash
  python MMFL_main.py
  ```

### 5. USC-HAD Dataset
- Navigate to the `USC-HAD` folder:
  ```bash
  cd USC-HAD
  ```
- Run the following command:
  ```bash
  python MMFL_main.py
  ```

## Shapley Value Implementation

The implementation of Shapley values is included in the aggregation section of the code. For example, in the `AD` folder, you can refer to lines 1011 to 1128 of the `MMFL_main.py` file for the corresponding implementation.

---

## New Experiments

### Feature Visualization

We explore the features extracted with and without the shared feature space method on the MHAD dataset. Specifically, we visualize the features of the individual modality (two modalities in total), before the fusion of modalities, and after the fusion of modalities. The visualization results are shown in the figure below.

- w/o shared feature space for modality 1 in the MHAD dataset

![avatar](https://github.com/yiliucs/WWW25/blob/main/MHAD-M1-Before.png)

- w/ shared feature space for modality 1 in the MHAD dataset

![avatar](https://github.com/yiliucs/WWW25/blob/main/MHAD-M1-After.png)

- w/o shared feature space for modality 2 in the MHAD dataset

![avatar](https://github.com/yiliucs/WWW25/blob/main/MHAD-M2-Before.png)

- w/ shared feature space for modality 2 in the MHAD dataset

![avatar](https://github.com/yiliucs/WWW25/blob/main/MHAD-M2-After.png)

- w/o shared feature space for fused modality in the MHAD dataset

![avatar](https://github.com/yiliucs/WWW25/blob/main/MHAD.png)
