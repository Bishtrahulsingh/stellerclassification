# Automated Celestial Object Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hXR-gJ1bZPVo6_3km7tuLo9CfAYVD5JS?usp=sharing)

## Project Overview

This project implements and evaluates a machine learning pipeline for the automated classification of celestial objects. Using a public dataset of 100,000 observations from a large-scale astronomical survey (likely the Sloan Digital Sky Survey, or SDSS), this work compares three different classification algorithms—Logistic Regression, Support Vector Machines (SVM), and Random Forest—to accurately categorize objects into **Galaxies**, **Quasars (QSOs)**, or **Stars**.

The final model (Random Forest) achieves **97.8% accuracy**, demonstrating the viability of using ensemble methods to reliably handle and classify large volumes of astronomical data.

## The Dataset

The dataset consists of 100,000 entries, each representing a unique celestial object. The classification is based on 8 key features.

* **Target Variable:** `class` - The object's classification (GALAXY, STAR, or QSO).
* **Key Features Used:**
    * `alpha`, `delta`: The celestial coordinates (Right Ascension and Declination) specifying the object's position on the sky.
    * `u`, `g`, `r`, `i`, `z`: The object's brightness (magnitude) as measured through five different photometric filters (ultraviolet, green, red, near-infrared, and infrared).
    * `redshift`: A measure of how much an object's light has been "stretched" by the expansion of the universe. It is a critical indicator of distance.

## Methodology

The project was structured as a formal machine learning pipeline, from data ingestion to model evaluation.

### 1. Data Preprocessing and Cleaning

1.  **Feature Selection:** Removed 9 non-physical metadata columns from the dataset. Columns like `obj_ID`, `run_ID`, `plate`, `MJD`, and `fiber_ID` are identifiers related to how and when the observation was taken, not intrinsic properties of the object itself.
2.  **Target Encoding:** The categorical `class` label was converted into a numerical format using `sklearn.preprocessing.LabelEncoder` (GALAXY: 0, QSO: 1, STAR: 2).
3.  **Train-Test Split:** The data was split into a 70% training set and a 30% testing set. A `stratify` parameter was used to ensure that the distribution of GALAXY, QSO, and STAR classes was identical in both the training and test sets, which is crucial for handling the dataset's natural imbalance.
4.  **Feature Scaling:** All 8 input features were scaled using `sklearn.preprocessing.StandardScaler`. This step is essential as features are on vastly different scales. Scaling centers all features to a mean of 0 and a standard deviation of 1, improving the performance and convergence of algorithms like Logistic Regression and SVM.

### 2. Comparative Model Analysis

Three distinct classification algorithms were trained on the preprocessed data to compare their effectiveness:

* **Logistic Regression:** Chosen as a simple, fast, and highly interpretable linear baseline model.
* **Support Vector Classifier (SVC):** A powerful model chosen for its ability to find complex, non-linear decision boundaries.
* **Random Forest Classifier:** An ensemble, tree-based model chosen for its high performance, robustness, and ability to capture complex, non-linear interactions between features.

## Results and Evaluation

The three trained models were evaluated on the unseen 30% test set. The results, as measured by classification accuracy, are as follows:

| Model | Test Accuracy |
| :--- | :--- |
| Logistic Regression | 95.94% |
| Support Vector Machine (SVC)| 95.96% |
| **Random Forest** | **97.80%** |

### Performance Analysis

The **Random Forest Classifier** demonstrated superior performance, yielding the highest accuracy (97.80%). This result significantly exceeds that of both the linear (Logistic Regression, 95.94%) and the kernel-based (SVC, 95.96%) models.

This performance disparity strongly indicates that the feature space is characterized by non-linear relationships. The classification of these celestial objects is evidently not a linearly separable problem. The Random Forest's ensemble of decision trees was uniquely effective at capturing the complex, high-order interactions between photometric colors and redshift. Conversely, the near-identical (and lower) scores of the Logistic Regression and default SVC models suggest that a simple linear decision boundary is insufficient to accurately model the inherent complexity of the data.

## Conclusion

This project successfully validated a machine learning pipeline for the automated classification of celestial objects. The high accuracy (97.8%) achieved by the Random Forest model confirms that modern, tree-based ensemble methods are exceptionally well-suited for this common astronomical research task. The model's ability to handle non-linear feature interactions proved to be the decisive factor, establishing a reliable methodology for processing and categorizing large-scale astronomical survey data.

## Technologies Used
* Python
* Pandas
* Numpy
* Scikit-learn
* Google Colab
