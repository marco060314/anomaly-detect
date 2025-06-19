import pandas as pd
import numpy as numpy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

class statistical_analysis:
    def zscore_outliers(df, col, threshold=3):
        z_scores = zscore(df[col])
        outliers = df[np.abs(z_scores) > threshold]
        return outliers
    
    def iqr_outliers(df):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower) | (df[column] > upper)]
        return outliers
    

    def detect_outliers_isolation_forest(df, contamination=0.05):
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        model = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly'] = model.fit_predict(numeric_df)
        return df[df['anomaly'] == -1]

    def find_outliers_from_fit(df, x_col, y_col, threshold=3):
        # Drop missing values
        df_clean = df[[x_col, y_col]].dropna()

        # Reshape for sklearn
        X = df_clean[[x_col]]
        y = df_clean[y_col]

        # Fit linear model
        model = LinearRegression()
        model.fit(X, y)

        # Predict y values
        y_pred = model.predict(X)

        # Calculate residuals (actual - predicted)
        residuals = y - y_pred
        df_clean['residual'] = residuals

        # Calculate z-score of residuals
        df_clean['z_resid'] = zscore(residuals)

        # Identify outliers
        outliers = df_clean[np.abs(df_clean['z_resid']) > threshold]

        return outliers, model