# House Price Prediction
![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-blueviolet)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Git](https://img.shields.io/badge/Git-Version%20Control-red)


House Price Prediction is an end-to-end machine learning project using the Ames Housing dataset to predict sale prices with reproducible pipelines and model comparison. The project implements robust preprocessing, cross-validation, and model selection to achieve significant performance gains.

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
- Final predictions are generated locally from the trained pipeline and are not versioned, as they are fully reproducible.

### Conclusion

- HistGradientBoosting captured non-linear patterns that linear models missed.


---

## Key Achievements

- Reduced log-RMSE from 0.148 (Ridge) to 0.134 (HistGradientBoosting)

- Built reproducible preprocessing pipelines with scikit-learn

- Compared multiple models and documented rationale

---


## Status

- Data loading ✅
- EDA ✅
- Preprocessing pipeline ✅
- Baseline model & CV evaluation ✅
- Advanced models (ElasticNet, tree-based models) ✅

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


