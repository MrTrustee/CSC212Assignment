#CSC ASSIGNMENT FOR ALL THE YEAR 2 STUDENTS.
# This is the program to access a webpage and download a Csv file from there.



import requests
import csv
import pandas as pd
import zipfile
import io
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def download_and_store_csv(url, filename="For_Prediction.csv", max_rows=201):
    """This function downloads a zip file from a URL, extracts the CSV, reads up to max_rows, and stores it.
       And also handles the exceptions and opens the file in text mode.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Since the file has to be downloaded from the url,
        # This might return exception so I tried to zip the file into memory
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        csv_filename = [f for f in zip_file.namelist() if f.endswith('.csv')][0]  # I used this program to get the CSV filename
        with zip_file.open(csv_filename) as csvfile_in:
            reader = csv.reader((line.decode('utf-8') for line in csvfile_in))
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile_out:
                writer = csv.writer(csvfile_out)
                row_count = 0
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
                    if row_count >= max_rows:
                        break

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
    except zipfile.BadZipFile as e:
        print(f"Error: Invalid zip file: {e}")
        return False
    except IndexError as e: #if no csv file is found in the zip file
        print(f"Error: No CSV file found in the zip archive: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    return True


def analyze_dataset(filename="For_Prediction.csv"):
    #This program open the required dataset and prints its rows and columns and the number of unique classes
    try:
        df = pd.read_csv(filename)
        rows, cols = df.shape
        last_column_name = df.columns[-1]
        num_classes = df[last_column_name].nunique()

        print(f"Number of rows: {rows}")
        print(f"Number of columns: {cols}")
        print(f"Number of unique classes in '{last_column_name}': {num_classes}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred during dataset analysis: {e}")
        return None


def predict_sales(df):
    """Uses linear regression to predict sales."""
    try:
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        return model

    except KeyError:
        print(f"Error: Target column '{target_column}' not found.")
        return None
    except ValueError:
        print("ValueError: Could not convert data to numeric.")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None


# This is where the Prediction start and execute the program
kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/jpallard/google-store-ecommerce-data-fake-retail-data" # Updated URL

if download_and_store_csv(kaggle_url):
    df = analyze_dataset()
    if df is not None:
        model = predict_sales(df)
        if model is not None:
            print("Linear Regression Model training and prediction successful")
else:
    print("Process failed.")