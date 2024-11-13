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


def plot_results(df, model, future_years=50):
    try:
        plt.figure(figsize=(15, 8))

        #plot historical data
        plt.scatter(df['Year'], df['J-D'],
                    color='#2ecc71',
                    alpha=0.5,
                    label='Historical Data')
        #plot trend line for historical data
        years = df['Year'].values.reshape(-1,1)
        plt.plot(df['Year'],
                model.predict(years),
                color='#2ecc71',
                linewidth=2,
                label='Historical Trend')
        
        #generate and plot future predictions
        last_year = int(df['Year'].max())
        future_x = np.array(range(last_year + 1, last_year + future_years + 1))
        future_predictions = model.predict(future_x.reshape(-1, 1))
       
        plt.plot(future_x, future_predictions,
                color='#e74c3c',
                linestyle='--',
                linewidth=2,
                label='Future Predictions')
       
        # Calculate and plot confidence interval
        mse = mean_squared_error(df['J-D'],
                               model.predict(years))
        std_dev = np.sqrt(mse)
        plt.fill_between(future_x,
                        future_predictions - 2*std_dev,
                        future_predictions + 2*std_dev,
                        color='#e74c3c',
                        alpha=0.2,
                        label='95% Confidence Interval')
       
        # Customize plot
        plt.title('Global Temperature Anomalies: Historical Data and Future Predictions',
                 fontsize=14, pad=20)
        plt.xlabel('Year')
        plt.ylabel('Temperature Anomaly (°C)')
        plt.legend()
       
        # Add trend information
        annual_trend = model.coef_[0]
        r2 = r2_score(df['J-D'], model.predict(years))
       
        info_text = f'Annual Trend: {annual_trend:.4f}°C/year\n'
        info_text += f'50-year projection: {(annual_trend * 50):.2f}°C\n'
        info_text += f'R² Score: {r2:.3f}'
       
        plt.text(0.02, 0.98, info_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top',
                fontsize=10)
       
        plt.tight_layout()
        plt.savefig('temperature_predictions.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'temperature_predictions.png'")
       
        return annual_trend, future_predictions[-1] - future_predictions[0]
   
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        raise

def main():
    try:
        # Load and process data
        print("Loading data...")
        df = load_and_clean_data('global_temperature.csv')
       
        # Create and evaluate model
        print("\nTraining model...")
        model, X_test, y_test, y_pred, mse, r2 = create_prediction_model(df)
       
        # Plot results and get predictions
        print("\nGenerating predictions and creating plot...")
        annual_trend, total_change = plot_results(df, model)
       
        # Print results
        print("\nModel Results:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"R² Score: {r2:.3f}")
        print(f"\nTemperature Trends:")
        print(f"Annual temperature change: {annual_trend:.4f}°C/year")
        print(f"Projected 50-year change: {total_change:.2f}°C")
       
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check the error message above and ensure all requirements are met.")


if __name__ == "__main__":
    main()






  
