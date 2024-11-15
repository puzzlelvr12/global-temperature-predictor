# Global Temperature Anomaly Prediction Model

This project develops a machine learning model to predict global temperature anomalies using historical data from NASA.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Model](#model)
- [Usage](#usage)

## Overview
Climate change is one of the most pressing issues facing the world today. This project aims to contribute to the understanding of global temperature trends by building a predictive model that can forecast future temperature anomalies.

The model is developed using Python and various data science libraries, including pandas, scikit-learn, and Matplotlib. A linear regression algorithm is trained on historical temperature data to generate forecasts up to 50 years into the future, along with a 95% confidence interval.

## Data
The dataset used in this project is obtained from the NASA Goddard Institute for Space Studies (GISS) Surface Temperature Analysis (GISTEMP). It contains monthly global temperature anomaly values from 2002 to 2023.

## Model
The model is built using the scikit-learn library's `LinearRegression` class. The key steps in the modeling process are:

1. Data preprocessing: Handling missing values, converting data types, and creating a clean dataset.
2. Train-test split: Dividing the data into training and testing sets to evaluate the model's performance.
3. Model training: Fitting the linear regression model to the training data.
4. Model evaluation: Assessing the model's performance using metrics such as R-squared and mean squared error.
5. Forecasting: Generating predictions for future years and calculating the 50-year temperature change.

## Usage
To use the model, you can clone the repository and run the `tempp_predictor.py` script. This will load the data, train the model, and generate the temperature predictions and visualization.

```bash
git clone https://github.com/puzzlelvr12/global-temperature-predictor.git
cd global-temperature-predictor
python -m venv tempp_predictor_env

tempp_predictor_env\Scripts\activate

pip install numpy pandas matplotlib scikit-learn

python tempp_predictor.py


