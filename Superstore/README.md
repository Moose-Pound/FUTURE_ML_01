# Superstore Sales Forecasting

An end-to-end time-series forecasting project that analyses historical Superstore sales and predicts future monthly sales using statistical and machine learning models.

The project combines exploratory data analysis, forecasting model development, walk-forward validation, and an interactive Streamlit dashboard.

**Live Demo:** [Moose Superstore Forecast](https://moose-superstore-forecast.streamlit.app/)

**GitHub Repository:** [FUTURE_ML_01](https://github.com/Moose-Pound/FUTURE_ML_01)

## Project Overview

This project uses the Sample Superstore dataset to investigate historical sales patterns from January 2014 to December 2017.

Four forecasting models are implemented and evaluated to determine which produces the most reliable predictions based on historical validation performance.

Key components include:

- Exploratory data analysis (EDA)
- Monthly sales trend analysis
- Time-series forecasting
- Expanding-window walk-forward validation
- Model performance comparison
- SARIMA residual diagnostics
- Interactive six-month sales forecasting

## Forecasting Models

| Model | Description |
|---|---|
| Linear Regression | Models the relationship between time and sales. |
| Holt-Winters | Captures trend and seasonal patterns. |
| Seasonal Naive | Uses sales from the corresponding period in the previous year. |
| SARIMA | Models time-series trends, seasonality, and dependencies. |

## Model Evaluation

Models are evaluated using expanding-window walk-forward validation.

| Validation Setting | Value |
|---|---:|
| Initial training period | 24 months |
| Forecast horizon | 3 months |
| Validation folds | 8 |
| Total validation observations | 24 |

### Walk-Forward Validation Results

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| SARIMA (Tuned) | 12,098.59 | 14,461.16 | 24.89% |
| Seasonal Naive | 13,993.90 | 17,224.28 | 24.94% |
| Holt-Winters | 12,706.81 | 15,483.76 | 25.76% |
| Linear Regression | 20,033.58 | 24,528.86 | 46.71% |

SARIMA achieved the lowest MAE, RMSE, and MAPE among the evaluated models. However, its MAPE advantage over Seasonal Naive was only 0.05 percentage points, suggesting a marginal improvement over the simpler baseline.

**Validation limitation:** SARIMA configurations were selected using the same walk-forward folds used in the final model comparison. This introduces potential model-selection optimism. An independent holdout period would provide a stronger estimate of generalisation performance.

### Model Diagnostics

SARIMA residuals were examined using autocorrelation analysis and the Ljung-Box test.

The test returned a p-value of 0.2910 at lag 12, providing insufficient evidence to reject the null hypothesis of no residual autocorrelation up to that lag.

Parameter stability checks identified estimation warnings and some unreliable standard errors. These limitations should be considered when interpreting the results.

## Interactive Dashboard

The Streamlit application provides three main sections.

### 1. Overview

- Historical sales performance
- Monthly sales trends
- Key sales statistics

### 2. Sales Forecast

- Select between Linear Regression, Holt-Winters, Seasonal Naive, and SARIMA.
- Generate six-month sales forecasts.
- Explore interactive forecast visualisations.
- Download forecast results.

### 3. Model Comparison

- Compare models using MAE, RMSE, and MAPE.
- Explore SARIMA tuning results.
- Examine actual versus predicted sales.
- Analyse model residuals.
- View autocorrelation diagnostics and Ljung-Box test results.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analysis and model development |
| Pandas | Data manipulation and time-series processing |
| NumPy | Numerical computations |
| Matplotlib | Data visualisation |
| Plotly | Interactive charts |
| Statsmodels | Statistical forecasting and diagnostics |
| Scikit-learn | Linear regression and evaluation metrics |
| Streamlit | Interactive dashboard |
| Joblib | Saving and loading trained models |
| Jupyter Notebook | Exploratory analysis and experimentation |

## Project Structure

```text
FUTURE_ML_01/
└── Superstore/
    ├── app.py
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   └── Sample - Superstore.csv
    ├── models/
    │   ├── linear_regression_model.pkl
    │   ├── holt_winters_model.pkl
    │   ├── seasonal_naive_model.pkl
    │   └── sarima_model.pkl
    └── notebooks/
        └── Superstore.ipynb
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Moose-Pound/FUTURE_ML_01.git
```

### 2. Navigate to the project directory

```bash
cd FUTURE_ML_01/Superstore
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The dashboard will be available at:

http://localhost:8501

## Future Improvements

- Evaluate models using an independent holdout period.
- Explore additional forecasting algorithms.
- Investigate external variables that may influence sales.
- Improve forecast accuracy through further experimentation.
- Incorporate new sales observations for ongoing model evaluation.
- Explore automated model monitoring and retraining.

## Author

**Moose Pound**

- GitHub: [Moose-Pound](https://github.com/Moose-Pound)
- Repository: [FUTURE_ML_01](https://github.com/Moose-Pound/FUTURE_ML_01)
- Live Application: [Moose Superstore Forecast](https://moose-superstore-forecast.streamlit.app/)