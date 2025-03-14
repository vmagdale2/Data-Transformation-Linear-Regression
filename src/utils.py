import pandas as pd
import numpy as np
import os
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv
from google.auth import default
import io


# Authenticate and Load .env Variables
def authenticate_and_load_env():
    auth.authenticate_user()  # <-- Added for stable authentication
    load_dotenv()
    creds, _ = default()
    service = build('drive', 'v3', credentials=creds)
    print("✅ Google Drive API authenticated successfully!")
    return service

# Test to see if it's saving.
# Load Data from Google Drive
def load_data_from_drive(service, file_id):  # This version expects **two arguments**
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        file_content.seek(0)
        return pd.read_csv(file_content)

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


# Data Cleaning
def clean_data(df):
    """
    Clean dataset by handling missing values, duplicates, and other common issues.

    Parameters:
    - df (DataFrame): Raw data

    Returns:
    - DataFrame: Cleaned data
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


# Data Transformation - Box-Cox Transformation
from scipy.stats import boxcox


def apply_boxcox(df, columns):
    """
    Apply Box-Cox transformation to specified columns.

    Parameters:
    - df (DataFrame): DataFrame with raw data.
    - columns (list): List of column names to apply Box-Cox transformation.

    Returns:
    - DataFrame: DataFrame with transformed columns
    """
    transformed_df = df.copy()
    for col in columns:
        if np.all(transformed_df[col] > 0):  # Box-Cox requires strictly positive values
            transformed_df[col], _ = boxcox(transformed_df[col])
        else:
            print(f"⚠️ Skipping {col}: Contains zero or negative values.")
    return transformed_df


# Feature Engineering
def add_new_features(df):
    """
    Add useful engineered features.

    Parameters:
    - df (DataFrame): DataFrame with clean data.

    Returns:
    - DataFrame: DataFrame with new features.
    """
    df['TAX_per_RM'] = df['TAX'] / df['RM']  # Example Feature
    df['AGE_BIN'] = pd.cut(df['AGE'], bins=[0, 30, 60, 100], labels=['Young', 'Middle-aged', 'Old'])
    return df


# Save Processed Data
def save_data(df, filename, folder='data/processed'):
    """
    Save processed data as CSV.

    Parameters:
    - df (DataFrame): Processed data
    - filename (str): File name to save
    - folder (str): Folder path for saving data
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    print(f"✅ Data saved successfully: {path}")
