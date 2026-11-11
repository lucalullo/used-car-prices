
# Used Car Prices

**Predicting used car prices with LightGBM - combining feature engineering, target transformation, and gradient boosting for tabular regression.**

This project implements an end-to-end machine learning workflow, from cleaning and transforming vehicle data to training and tuning a LightGBM regression model.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lucalullo/used-car-prices)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data_Processing-150458?logo=pandas&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-02569B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model_Selection-F7931E?logo=scikitlearn&logoColor=white)

**[View the original Kaggle notebook](https://www.kaggle.com/code/lucalullo/used-car-prices)** | **[Explore the notebook](used-car-prices.ipynb)** | **[Author's GitHub](https://github.com/lucalullo)**

---

## At a glance

| | |
|---|---|
| **Task** | Regression: used car price prediction |
| **Platform** | Kaggle |
| **Model** | LightGBM Regressor |
| **Approach** | Gradient boosting with feature engineering and target transformation |
| **Feature processing** | Missing values, text standardization, numerical transformations, categorical encoding |
| **Target transformation** | `log1p` for training, `expm1` for inverse transformation |
| **Model selection** | Hyperparameter tuning and early stopping |
| **Core libraries** | pandas, NumPy, scikit-learn, LightGBM |
| **Environment** | Python and Jupyter Notebook |

## Why this project matters

This is a **feature-engineering-focused regression experiment**: the emphasis is on converting heterogeneous vehicle information into useful predictive features and applying gradient boosting to estimate used car prices.

The notebook covers:

- **Data cleaning:** handling missing values, standardizing text fields, and extracting structured information from raw vehicle data.
- **Feature engineering:** creating additional variables such as vehicle age (`age`) and horsepower (`hp`).
- **Numerical transformations:** applying logarithmic transformations to skewed numerical features.
- **Categorical encoding:** preparing categorical vehicle attributes for regression modeling.
- **Target transformation:** using `log1p` on sale prices and `expm1` to return predictions to the original price scale.
- **Gradient boosting:** training a LightGBM Regressor with hyperparameter tuning and early stopping.

## Approach

```text
Used car dataset
        |
        v
Data cleaning and text standardization
        |
        v
Feature engineering
        |
        v
Numerical transformations
        |
        v
Categorical encoding
        |
        v
Target transformation (log1p)
        |
        v
LightGBM regression
        |
        v
Hyperparameter tuning + early stopping
        |
        v
Inverse transformation (expm1)
        |
        v
Predicted car prices
```

### Implementation details

| Component | Notebook approach |
|---|---|
| Missing values | Data cleaning and missing value handling |
| Text features | Standardization and extraction of structured information |
| Engineered features | Vehicle age (`age`) and horsepower (`hp`) |
| Numerical features | Logarithmic transformations for skewed variables |
| Categorical features | Encoding of vehicle attributes |
| Target | Logarithmic transformation with `log1p` |
| Model | LightGBM Regressor |
| Model optimization | Hyperparameter tuning and early stopping |
| Prediction output | Inverse transformation with `expm1` |

The workflow focuses on a single gradient boosting approach rather than a systematic comparison of multiple regression algorithms.

## Modeling approach

The project uses **LightGBM**, a gradient boosting framework designed for efficient learning on structured tabular data.

A central modeling choice is the logarithmic transformation of the target variable:

- **Training:** `log1p(price)` transforms the target before model fitting.
- **Prediction:** `expm1(prediction)` converts model outputs back to the original price scale.

This approach is intended to handle the skewed distribution commonly found in price data. Its effectiveness depends on the dataset and validation results.

Hyperparameter tuning and early stopping are used to control model complexity and reduce the risk of overfitting.

[Explore the original Kaggle notebook](https://www.kaggle.com/code/lucalullo/used-car-prices).

## Repository contents

```text
Used-car-prices/
├── README.md
└── used-car-prices.ipynb
```

The notebook contains the original data preparation, feature engineering, target transformation, and LightGBM modeling workflow.

## Run the notebook

The reference project is available on **Kaggle**, where the notebook can be explored in its original environment.

To run it locally, install the required dependencies:

```bash
python -m pip install pandas numpy scikit-learn lightgbm matplotlib seaborn jupyter
```

Then open `used-car-prices.ipynb` in Jupyter and make the required dataset available.

**Local execution may require updating dataset paths inside the notebook** to match the location of the downloaded files.

## Scope and limitations

- The project focuses on **LightGBM regression**, not a systematic comparison against alternative machine learning algorithms.
- Feature engineering and preprocessing decisions are tailored to the available vehicle data.
- The logarithmic target transformation changes the scale on which the model is trained; predictions require inverse transformation to recover price estimates.
- Hyperparameter tuning and early stopping do not eliminate the possibility of overfitting.
- Performance on different vehicle markets or datasets is not established by the original project description.
- No numerical Kaggle evaluation score is documented in the supplied README.

## Related work

This project complements my other work on applied machine learning, feature engineering, and predictive modeling with Python.

- [GitHub profile and selected projects](https://github.com/lucalullo)
- [Kaggle profile](https://www.kaggle.com/lucalullo)

---

**Author:** [Luca Lullo](https://github.com/lucalullo)  
**Original Kaggle work:** [Used Car Prices](https://www.kaggle.com/code/lucalullo/used-car-prices)
