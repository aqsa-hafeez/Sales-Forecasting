# Walmart Sales Forecasting 

## Overview
This project focuses on forecasting weekly sales for Walmart stores using historical sales data. The dataset is sourced from the [Walmart Sales Forecasting competition on Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting). The goal is to predict future sales by leveraging time-series features, regression models, and visualizations to compare actual vs. predicted values.

Key components:
- **Data Processing**: Merging multiple CSV files (train.csv, features.csv, stores.csv) into a single dataframe.
- **Feature Engineering**: Creating time-based features (e.g., year, month, week, day), lag features (e.g., previous week's sales), and rolling averages.
- **Modeling**: Training regression models like RandomForestRegressor and XGBoost for sales prediction.
- **Evaluation**: Using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Visualization**: Plotting actual vs. predicted sales over time for validation data (2012 period).
- **Advanced Topics**: Incorporates seasonal decomposition, moving averages, and time-aware validation using XGBoost or LightGBM (though LightGBM is mentioned in the task description, XGBoost is used in the code).

This notebook demonstrates time-series forecasting techniques, regression modeling, and data visualization using Python libraries.

## Dataset
- **Recommended Dataset**: Walmart Sales Forecast from Kaggle.
- Files Used:
  - `train.csv`: Historical weekly sales data by store and department.
  - `features.csv`: Additional features like temperature, fuel price, CPI, unemployment, and markdowns.
  - `stores.csv`: Store details like type and size.
- **Note**: The code assumes these files are located in the directory `C:\Users\hafee\Videos\Walmart_sales_forecast\`. Update the file paths in the notebook if your files are stored elsewhere. Download the dataset from Kaggle and place the files in the appropriate directory.

## Requirements
- Python 3.10+ (tested on 3.10.0)
- Libraries (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - statsmodels
  - xgboost

### Installation
1. Clone or download this repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the Walmart dataset from Kaggle and place the CSV files in the specified path (or update the paths in the notebook).

## How to Run
1. Open the Jupyter notebook `code1.ipynb` in Jupyter Lab or Jupyter Notebook:
   ```
   jupyter notebook code1.ipynb
   ```
2. Run the cells sequentially:
   - Cell 1: Import libraries.
   - Cell 2: Load and merge data (ensure CSV files are available).
   - Cell 3: Data preprocessing and feature engineering.
   - Cell 4-7: Define features, split data (time-based split: train up to 2011, validate 2012).
   - Cell 8: Train RandomForestRegressor and evaluate.
   - Cell 9: Train XGBoost and evaluate.
   - Cell 10: Plot actual vs. predicted sales (RandomForest).
   - Cell 11: Plot actual vs. predicted sales (XGBoost).
   - Cell 12: Combined plot of actual vs. both models' predictions.
3. Outputs include:
   - Printed shapes of dataframes.
   - Model performance metrics (MAE, RMSE).
   - Plots of sales forecasts.

## Key Features and Techniques
- **Time-Based Features**: Extracted year, month, week, and day from dates.
- **Lag and Rolling Features**: 1-week lag of sales and 4-week rolling average for capturing trends.
- **Categorical Encoding**: One-hot encoding for store types (A, B, C).
- **Time-Series Split**: Training on data up to 2011-12-31, validation on 2012 data to simulate real-world forecasting.
- **Models**:
  - RandomForestRegressor: Ensemble tree-based model for handling non-linear relationships.
  - XGBoost: Gradient boosting model with time-aware validation (supports early stopping).
- **Bonus Techniques**:
  - Seasonal decomposition using `statsmodels.tsa.seasonal.seasonal_decompose`.
  - Handling missing values (e.g., filling markdowns with 0).
- **Plots**: Matplotlib visualizations comparing actual and predicted sales aggregated by date.

## Results
- Model performance is evaluated on the validation set (2012 data).
- Example metrics (from a sample run; results may vary):
  - RandomForest: MAE ~2000-3000, RMSE ~4000-5000 (depending on hyperparameters).
  - XGBoost: Often outperforms RandomForest with lower errors due to boosting.
- Visualizations show how well predictions align with actual sales trends, including seasonal peaks.

## Limitations and Improvements
- **Data Path Dependency**: Hardcoded paths; consider using relative paths or environment variables.
- **Hyperparameter Tuning**: Not implemented; use GridSearchCV or Optuna for better results.
- **Advanced Models**: The task mentions LightGBM or XGBoost; this code uses XGBoost, but LightGBM could be swapped in for faster training.
- **Scalability**: For larger datasets, consider distributed training or sampling.
- **No External Data**: Relies solely on provided features; adding holidays or economic indicators could improve accuracy.
- **Error Handling**: Basic file not found check; expand for robustness.

## License
This project is for educational purposes. The code is open-source under the MIT License. Dataset usage follows Kaggle's terms.

## Contact
For questions or contributions, feel free to open an issue or pull request in the repository.
