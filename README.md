# Used Car Prices

**Used car price prediction with LightGBM, engineered vehicle features, and a logarithmic target transformation.**

This Kaggle notebook develops a tabular regression workflow, from cleaning vehicle specifications to validating a gradient-boosted model and generating competition predictions.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-Regression-02569B) [![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lucalullo/used-car-prices)

[Kaggle notebook](https://www.kaggle.com/code/lucalullo/used-car-prices) · [GitHub repository](https://github.com/lucalullo/used-car-prices)

## At a glance

| | |
|---|---|
| **Task** | Regression: used car selling price |
| **Competition** | Kaggle Playground Series, Season 4 Episode 9 |
| **Model** | LightGBM Regressor |
| **Validation** | 80/20 train/validation split; mean absolute error (MAE) on original price scale; target encoding fitted before the split |
| **Recorded validation MAE** | **17,374** (notebook output; in the target's price units) |
| **Feature engineering** | Vehicle age, engine horsepower, displacement, cylinder count, categorical attributes |
| **Target** | `log1p` for model fitting; `expm1` for predictions |
| **Core libraries** | pandas, NumPy, scikit-learn, LightGBM |
| **Data** | Kaggle competition train/test CSVs (not redistributed) |

## Project overview

This is a **feature-engineering-focused regression experiment**. Vehicle descriptions are converted into structured signals, missing numerical values are imputed, and LightGBM learns nonlinear relationships between vehicle characteristics and prices.

The notebook covers:

- **Data preparation:** standardizing fields such as accident history and title status.
- **Feature engineering:** extracting horsepower, engine displacement and cylinder count, and computing vehicle age.
- **Missing values:** imputing selected numerical features with `IterativeImputer`.
- **Encoding and transformations:** target-mean encoding categorical variables using the full labeled dataset and applying logarithmic transformations to selected numerical variables.
- **Model training:** fitting LightGBM on log-transformed prices with validation-based early stopping.
- **Submission:** refitting on the full training set and exporting `submission.csv`.

## Methodology

~~~text
Kaggle train/test data
        |
        v
Cleaning + vehicle feature extraction
        |
        v
Missing-value imputation + target encoding
(using all labeled rows)
        |
        v
Numerical and target transformations
        |
        v
Train/validation split
        |
        v
LightGBM + early stopping
        |
        v
Validation MAE on original price scale
        |
        v
Refit on all labeled rows
        |
        v
Predict test set -> submission.csv
~~~

### Implementation details

| Component | Notebook implementation |
|---|---|
| Validation | `train_test_split(test_size=0.2, random_state=42)` |
| Imputation | `IterativeImputer` for selected numerical features, fitted before the validation split |
| Categorical encoding | Target-mean encoding computed from all labeled rows before the split |
| Base model | `LGBMRegressor(max_depth=10, learning_rate=0.008, n_estimators=8000, min_child_samples=50)` |
| Early stopping | 50 rounds without validation improvement |
| Final model | Refit on all labeled rows with number of trees derived from the best iteration |
| Evaluation | MAE after `expm1` converts predictions and targets to the original scale |

The notebook uses manually specified model settings and early stopping. It **does not demonstrate a systematic hyperparameter search** or comparative benchmark against other regressors.

## Results

> **Recorded validation MAE: 17,374** (in the original target units).

This value is printed in the archived notebook after an 80/20 holdout evaluation. **It should not be treated as an unbiased validation estimate:** category-wise target means are computed on all labeled rows before splitting, allowing validation targets to influence encoded validation features. The imputer is also fitted before splitting. It is **not a Kaggle leaderboard score** and does not establish performance on other used car markets.

## Repository structure

~~~text
used-car-prices/
├── README.md
├── used-car-prices.ipynb
├── .gitignore
└── LICENSE
~~~

## Getting started

The reference environment is **Kaggle**. The notebook reads the competition files from:

~~~text
/kaggle/input/competitions/playground-series-s4e9/
~~~

For local execution, install the principal notebook dependencies:

~~~bash
python -m pip install pandas numpy scikit-learn lightgbm matplotlib seaborn jupyter
~~~

Download the competition train/test CSVs, open `used-car-prices.ipynb`, and update the Kaggle-specific input paths to the local data directory. The notebook writes a `submission.csv` file.

## Limitations

- This repository preserves a notebook experiment, not a packaged prediction service.
- **Target leakage in validation:** target-mean encoding uses labels from the entire training dataset before the holdout split. The reported MAE may therefore be optimistic. A leakage-safe evaluation would fit the encoder only on training folds (ideally using out-of-fold encodings) and apply it to the holdout without using holdout labels.
- The numerical imputer is also fitted before the holdout split; preprocessing should be fitted inside the training pipeline for an independent evaluation.
- The reported MAE is from a single holdout split; no cross-validation or independent external test is provided.
- Model settings are chosen manually rather than through an automated hyperparameter search.
- The preprocessing workflow is specific to the competition dataset and may need adaptation elsewhere.
- The competition data are not redistributed; access and permitted use are governed by Kaggle.

## Related work

This analysis complements my work on tabular machine learning, from-scratch algorithms, and reproducible Python projects.

**[GitHub profile](https://github.com/lucalullo) · [Kaggle profile](https://www.kaggle.com/lucalullo)**

## Author

Created by [Luca Lullo](https://github.com/lucalullo).