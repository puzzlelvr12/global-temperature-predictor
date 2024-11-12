import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Set plotting style using built in matplotlib style
plt.style.use('default')#
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_and_clean_data(file_path):
    try:
        #read the CSV file, skipping the first row (header)
        df = pd.read_csv(file_path, skiprows=[0])

        #convert Year to numeric only
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        #convert '*******' to NaN 
        df = df.replace('*******', np.nan)

        #convert all columns except 'Year' to float
        for col in df.columns:
            if col != 'Year':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
    
        #creates copy df with only Year and J-D
        df = df[['Year', 'J-D']].copy()

        #drop rows with missing values 
        df = df.dropna()

        #ensures Year is type int 
        df['Year'] = df['Year'].astype(int)

        print("Data loaded and cleaned successfully")
        print(f"Number of valid records: {len(df)}")
        print("\nFirst few rows of cleaned data:")
        print(df.head())
        print("\nData summary")
        print(df.describe())

        return df
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
        print("Make sure the CSV file is in the same directory as the script")
        raise 
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

def create_prediction_model(df):
    #prepare data for modeling
    X = df['Year'].values.reshape(-1,1)
    y = df['J-D'].values

    #split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #create and train the model 
    model = LinearRegression()
    model.fit(X_train, y_train)

    #make predictions on test set
    y_pred = model.predict(X_test)

    #calculate metrics and evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel training completed: ")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    return model, X_test, y_test, y_pred, mse, r2


