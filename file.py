import os
from typing import Tuple

import pandas as pd


def get_file_name(file_path: str) -> str:
    """
    Extract the file name from a given file path.

    Parameters:
    - file_path: str - The path to the file.

    Returns:
    - str - The file name without the directory path.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    return os.path.basename(file_path)


def get_file_name_without_extension(file_path: str) -> str:
    """
    Extract the file name without extension from a given file path.

    Parameters:
    - file_path: str - The path to the file.

    Returns:
    - str - The file name without the directory path and extension.
    """
    file_name = get_file_name(file_path)
    return os.path.splitext(file_name)[0]


def extract_file_details(file_path: str) -> Tuple[str, str]:
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    file_name = get_file_name_without_extension(file_path)
    parts = file_name.split('_')
    if len(parts) != 3:
        raise ValueError("file_name must be in the format 'prefix_participant_trial'")
    _, participant_code, trial_number = parts
    return participant_code, trial_number


def read_csv_file(file_path: str) -> pd.DataFrame:
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    return pd.read_csv(file_path)


def check_dir(dir_path: str) -> None:
    """
    Ensure that the specified directory exists; if not, create it.

    Parameters:
    directory (str): The directory path.

    Returns:
    None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")
