import numpy as np
import pandas as pd


def calculate_mean(df: pd.Series) -> float:
    """
    Calculate the mean of a given Series.

    Parameters:
    - df: pd.Series - The Series containing the data.

    Returns:
    - float - The mean value of the Series.
    """
    if not isinstance(df, pd.Series):
        raise TypeError("df must be a pandas Series")

    return df.mean()


def calculate_std(df: pd.Series) -> float:
    """
    Calculate the standard deviation of a given Series.

    Parameters:
    - df: pd.Series - The Series containing the data.

    Returns:
    - float - The standard deviation of the Series.
    """
    if not isinstance(df, pd.Series):
        raise TypeError("df must be a pandas Series")

    return df.std()


def find_max_index(series: pd.Series) -> int:
    """
    Find the index of the maximum value in a given Series.

    Parameters:
    - series: pd.Series - The Series to find the maximum value in.

    Returns:
    - int - The index of the maximum value.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    return series.idxmax()


def find_max_value(series: pd.Series) -> float:
    """
    Find the maximum value in a given Series.

    Parameters:
    - series: pd.Series - The Series to find the maximum value in.

    Returns:
    - float - The maximum value.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    return series.max()


def slice_dataframe(df, start_index=None, end_index=None):
    """
    Slice the DataFrame from the specified start index to the end index.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    start_index (int or None): The start index for slicing. If None, slicing starts from the beginning.
    end_index (int or None): The end index for slicing. If None, slicing goes until the end.

    Returns:
    pd.DataFrame: A subset of the input DataFrame from start_index to end_index.
    """
    if start_index is None:
        start_index = df.index[0]
    if end_index is None:
        end_index = df.index[-1]

    subset_df = df.loc[start_index:end_index]
    return subset_df


def subset_from_max(df: pd.DataFrame, force_column: str='force'):
    """
    Subset the DataFrame from the index of the maximum force value until the end.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    force_column (str): The name of the column containing force values. Default is 'force'.

    Returns:
    pd.DataFrame: A subset of the input DataFrame from the maximum force value index to the end.
    """
    # Step 1: Identify the index of the maximum force value
    max_force_index = find_max_index(df[force_column])

    # Step 2: Slice the DataFrame from this index until the end
    subset_df = slice_dataframe(df, max_force_index)

    return subset_df


def add_custom_column(df: pd.DataFrame, column_name: str, values: list=None):
    """
    Add a new column to the DataFrame with custom specified values or a default value.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the new column to add.
        values (list, array-like, optional): The values to be added to the new column.
                                            If None, fill the column with default_value. Default is None.

    Returns:
        pd.DataFrame: The DataFrame with the new column added.
    """
    if values is None:
        df[column_name] = np.arange(len(df))
    else:
        if len(values) != len(df):
            raise ValueError("Length of 'values' must be equal to the length of the DataFrame")
        df[column_name] = values
    return df


def extract_column_values(df: pd.DataFrame, column_name: str) -> np.ndarray:
    """
    Extract values of a specific column from a DataFrame as a numpy array.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column whose values are to be extracted.

    Returns:
    np.ndarray: A numpy array containing the values of the specified column.

    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    TypeError: If the input arguments are not of the expected types.
    """
    # Check if column_name exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    return df[column_name].values


def rename_column(_df: pd.DataFrame, col_num: int = 0, new_name: str = 'force', inplace: bool = True):
    if not isinstance(_df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(col_num, int):
        raise TypeError("col_num must be an integer")
    if not isinstance(new_name, str):
        raise TypeError("new_name must be a string")
    if not isinstance(inplace, bool):
        raise TypeError("inplace must be a boolean")
    _df.rename(columns={_df.columns[col_num]: new_name}, inplace=inplace)


def save_dataframe_to_csv(_df: pd.DataFrame, _output_file_path: str) -> None:
    """
    Save DataFrame to a CSV file and print a message.

    Parameters:
        _df (pd.DataFrame): The DataFrame to save.
        _output_file_path (str): The file path where the DataFrame will be saved.

    Returns:
        None
    """
    _df.to_csv(_output_file_path, index=False)
    print(f"Output saved to {_output_file_path}")
