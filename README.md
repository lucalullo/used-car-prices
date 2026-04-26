# Used Car Prices

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![LightGBM](https://img.shields.io/badge/lightgbm-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

This repository contains a complete data science workflow for predicting used car prices using machine learning.

🔗 **Original Notebook on Kaggle:** [View here](https://www.kaggle.com/code/lucalullo/used-car-prices)  
📓 **Notebook on GitHub:** [View here](https://github.com/lucalullo/Used-car-prices/blob/main/used-car-prices.ipynb)

## 📈 Performance
- **Model:** LightGBM Regressor  
- **Approach:** Gradient Boosting with feature engineering and target transformation  

## 🛠️ Key Features
- **Data Cleaning:**  
  - Handling missing values  
  - Standardization of text features  
  - Extraction of structured information from raw data  

- **Feature Engineering:**  
  - Creation of new features such as `age` and `hp`  
  - Log transformation of skewed variables  
  - Encoding of categorical features  

- **Target Transformation:**  
  - Log transformation (`log1p`) applied to price  
  - Inverse transformation using `expm1`  

- **Modeling:**  
  - LightGBM implementation for tabular data  
  - Early stopping to prevent overfitting  
  - Hyperparameter tuning  

## 🚀 How to use
1. Clone the repo: `git clone https://github.com/lucalullo/Used-car-prices.git`  
2. Install dependencies: `pip install pandas numpy scikit-learn lightgbm matplotlib seaborn`  
3. Run the `used-car-prices.ipynb` notebook  

---

**Author:** [Luca Lullo](https://github.com/lucalullo)  
*Data Scientist | Machine Learning Applied*
