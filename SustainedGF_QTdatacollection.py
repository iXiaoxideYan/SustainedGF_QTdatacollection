# Sustained Grip force
# From Yan's QT program

# Import packages and classes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from df import find_max_index, find_max_value, calculate_mean, calculate_std, slice_dataframe, add_custom_column, \
    extract_column_values
from file import read_csv_file, extract_file_details, check_dir

# File and Dir path
path = './rawdata_33_33.csv'
output_directory = './output'


def read_data(file_path: str):
    _df = read_csv_file(file_path)
    _participant_code, _trial_number = extract_file_details(file_path)
    return _df, _participant_code, _trial_number


# df, participant_code, trial_number = read_data(path)


# Rename the first column to 'force'
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


def plot_raw_data(_df: pd.DataFrame, label: str = 'Force-Time Curve'):
    plt.plot(_df, label=label)
    plt.title('Force-Time Curve Raw data')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.show()


# 1: HAUC
def calculate_hauc(_max_force_value: float, df_length: int) -> float:
    """
    Calculate HAUC (Highest Average Usable Consumption) based on the maximum force value
    and the number of samples.

    Parameters:
        _max_force_value (float): The maximum force value.
        df_length (int): The number of samples in the DataFrame.

    Returns:
        float: The calculated HAUC value.
    """
    _hauc = _max_force_value * df_length
    return _hauc


# 2: AUC
def calculate_auc(time: np.ndarray, force: np.ndarray) -> float:
    """
    Calculate AUC using the trapezoidal rule.

    Parameters:
        time (np.ndarray): Array of time values.
        force (np.ndarray): Array of force values.

    Returns:
        float: The calculated AUC value.
    """
    _auc = np.trapz(force, time)
    return _auc


# 3: Calculate the SFI [100% x 1 - (AUC/HAUC)]
def calculate_sfi(_auc: float, _hauc: float) -> float:
    """
    Calculate SFI as a percentage.

    Parameters:
        _auc (float):
        _hauc (float):

    Returns:
        float: The calculated SFI percentage.
    """
    _sfi = 100 * (1 - (_auc / _hauc))
    return _sfi


# Create AUC Plot
def plot_auc(x_time: np.ndarray, y_force: np.ndarray, _max_force_value: float, _auc: float, _sfi: float,
             output_file: str = None):
    """
    Plot the force-time curve with AUC and related annotations.

    Parameters:
        x_time (np.ndarray): Array of time values.
        y_force (np.ndarray): Array of force values.
        _max_force_value (float): Maximum force value.
        _auc (float): Area under the force-time curve (AUC).
        _sfi (float): Skeletal Force Index (SFI) percentage.
        output_file (str, optional): File path to save the plot. If None, plot is displayed without saving.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    # Plot force-time curve
    plt.plot(x_time, y_force, label='Force-Time Curve')

    # Plot area between max force and AUC
    max_line = np.full_like(x_time, _max_force_value)
    plt.fill_between(x_time, max_line, y_force, where=(y_force <= max_line), alpha=0.5,
                     label='Area Between Max Force and AUC')

    # Plot max force line
    plt.axhline(y=_max_force_value, color='r', linestyle='--', label='Max Force Value')

    # Annotations and plot settings
    plt.title('Force-Time Curve')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.text(0.05, 0.05, 'AUC', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom',
             horizontalalignment='left')
    plt.text(0.95, 0.95, 'HAUC', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             horizontalalignment='right')
    plt.text(0.95, 0.95, f'SFI: {_sfi:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             horizontalalignment='right')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()

    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


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


if __name__ == '__main__':
    # Data Input and General Shift
    df, participant_code, trial_number = read_data(path)
    rename_column(df, col_num=0, new_name='force')

    # Plot Raw Data
    plot_raw_data(df)

    # Find the index and the value of the maximum value in the 'force' column
    series = df['force']
    max_force_index = find_max_index(series)
    max_force_value = find_max_value(series)
    print(f"Maximum Force occurs at: {max_force_index} sample")
    print(f"Maximum Force is: {max_force_value}")

    # Subset the DataFrame from the index of the maximum force value until the end
    df_subset = slice_dataframe(df, max_force_index)

    df_subset = add_custom_column(df_subset, 'time')

    # Mean Force
    sub_series = df_subset['force']
    mean_force_value = calculate_mean(sub_series)
    sd_force_value = calculate_std(sub_series)

    # Extract time and force values
    time_values = extract_column_values(df_subset, 'time')
    force_values = extract_column_values(df_subset, 'force')

    # Calculate HAUC
    hauc = calculate_hauc(max_force_value, len(df))
    print(f"The number of samples in the DataFrame is: {len(df)}")
    print(f"The HAUC is: {hauc}")

    # Calculate AUC
    auc = calculate_auc(time_values, force_values)
    print(f"Area under the force-time curve (AUC): {auc}")

    # Calculate SFI
    sfi = calculate_sfi(auc, hauc)
    print(f"Skeletal Force Index (SFI): {sfi:.2f}%")

    # Plot the force-time curve
    plot_auc(time_values, force_values, max_force_value, auc, sfi)

    df_output = pd.DataFrame({
        'Participant': [participant_code],
        'Trial': [trial_number],
        'max_force': [max_force_value],
        'mean_force': [mean_force_value],
        'sd_force': [sd_force_value],
        'auc': [auc],
        'sfi': [sfi]
    })

    # Save the DataFrame to a CSV file
    output_file_name = f"{participant_code}_{trial_number}_susgf_vars.csv"
    # Full path to the output file
    output_file_path = os.path.join(output_directory, output_file_name)

    # Ensure the output directory exists
    check_dir(output_directory)

    # Save the DataFrame to the specified CSV file
    save_dataframe_to_csv(df_output, output_file_path)