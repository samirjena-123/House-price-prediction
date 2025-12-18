# House Price Prediction

End-to-end machine learning project using structured tabular data to predict house prices.
The project emphasizes **clean preprocessing pipelines**, **reproducibility**, and **proper model evaluation**.

---

## Problem Statement

Predict house sale prices (`SalePrice`) using structured tabular features  
(supervised regression).

---

## Dataset

**Ames Housing Dataset**

- ~1,460 training samples
- 80 input features
- Mix of numerical and categorical variables
- Strongly right-skewed target distribution

---

## Project Structure

```
data/
└── raw/ # Original dataset files

notebooks/
├── 01_EDA.ipynb # Exploratory data analysis
└── 02_preprocessing_pipeline.ipynb # Preprocessing, pipelines, baseline models

requirements.txt # Python dependencies
README.md
```
---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Target distribution analysis
- Missing value analysis
- Numerical vs categorical feature split
- Correlation analysis
- Neighborhood-based price analysis

### 2. Preprocessing
- Numerical features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - Constant imputation (`"None"`)
  - Ordinal encoding with unknown handling
- Implemented using **scikit-learn Pipelines and ColumnTransformer**


### 3. Modeling
- Baseline model: **Ridge Regression**
- Target trained in **log space** (`log1p(SalePrice)`)
- Evaluation using **5-fold cross-validation**
- Metric: **RMSE (log space)**
- Target variable transformed using `log1p(SalePrice)` to handle skewness

---

## Results

- Baseline Ridge CV log-RMSE ≈ **0.148 ± 0.041**
- Log-transforming the target significantly improves performance
- Pipeline is stable and reproducible

---

## Status

- Data loading ✅
- EDA ✅
- Preprocessing pipeline ✅
- Baseline model & CV evaluation ✅
- Advanced models (ElasticNet, tree-based models) ⏳

---

## How to Run

```bash
git clone <repo-url>
cd House-price-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Open notebooks in order:
1. `01_EDA.ipynb`
2. `02_preprocessing_pipeline.ipynb`


