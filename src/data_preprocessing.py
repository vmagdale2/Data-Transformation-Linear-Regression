import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats

def handle_missing_values(df, strategy='mean'):
    """
    Handles missing values in the dataset.
    - `strategy`: 'mean', 'median', 'mode', or 'drop'.
    """
    imputer = SimpleImputer(strategy=strategy)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def remove_duplicates(df):
    """
    Removes duplicate rows from the dataset.
    """
    return df.drop_duplicates().reset_index(drop=True)

def detect_outliers_iqr(df, threshold=1.5):
    """
    Detects outliers using the IQR method.
    """
    outliers = pd.DataFrame()
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))
        outliers[col] = mask
    return outliers

def transform_outliers(df, method='log', features=None):
    """
    Transforms outliers instead of removing them.
    Supported methods: 'log', 'sqrt', 'boxcox', 'yeojohnson'.
    """
    transformed_df = df.copy()
    for col in features if features else df.select_dtypes(include=[np.number]).columns:
        if method == 'log':
            transformed_df[col] = np.log1p(transformed_df[col])
        elif method == 'sqrt':
            transformed_df[col] = np.sqrt(transformed_df[col])
        elif method == 'boxcox':
            transformed_df[col], _ = stats.boxcox(transformed_df[col] + 1)
        elif method == 'yeojohnson':
            pt = PowerTransformer(method='yeo-johnson')
            transformed_df[col] = pt.fit_transform(transformed_df[[col]])
    return transformed_df

def scale_features(df, method='standard'):
    """
    Scales numerical features using the chosen method.
    Supported methods: 'standard', 'minmax', 'robust'.
    """
    scaler = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }.get(method, StandardScaler())

    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def encode_categorical(df):
    """
    Encodes categorical features using one-hot encoding.
    """
    return pd.get_dummies(df, drop_first=True)

def preprocess_data(df):
    """
    Master function to run all preprocessing steps in order.
    """
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    outliers = detect_outliers_iqr(df)
    transformed_df = transform_outliers(df, method='log', features=outliers.columns[outliers.any()])
    transformed_df = scale_features(transformed_df, method='standard')
    return transformed_df
