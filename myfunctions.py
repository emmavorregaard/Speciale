import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def transform(series, code):
    """
    Function to map transformation codes to functions
    """
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Logarithm
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # First difference of log minus lagged first difference of log
        return np.log(series).diff() - np.log(series).diff().shift(1)
    elif code == 7:
        # First difference of the rate of change
        return (series / series.shift(1) - 1).diff()
    else:
        raise ValueError(f"Unknown transformation code: {code}")
    
def apply_transformations(df, value_column='value', tcode_column='TCODE'):
    """
    Function to transform columns in the data
    """
    # Create an empty DataFrame to store the transformed values
    transformed_df = pd.DataFrame()

    # Group by the 'variable' column and apply the transformation for each group
    for variable, group in df.groupby('variable'):
        # Get the transformation code for this variable (assuming it's constant within the group)
        tcode = group[tcode_column].iloc[0]
        # Apply the transformation based on the tcode
        group['transformed_value'] = transform(group[value_column], tcode)
        # Append the transformed group to the transformed_df DataFrame
        transformed_df = pd.concat([transformed_df, group])

    return transformed_df

def test_for_unit_roots_5pct(df, variable_column, transformed_value_column):
    results = []
    for variable, group in df.groupby(variable_column):
        time_series = group[transformed_value_column].dropna()  # Ensure there are no NaNs
        adf_result = adfuller(time_series)
        test_statistic = adf_result[0]
        critical_value_5pct = adf_result[4]['5%']
        p_value = adf_result[1]
        # Check if the test statistic is less than the critical value at 5%
        reject_null = test_statistic < critical_value_5pct
        results.append({
            'variable': variable,
            'ADF Statistic': test_statistic,
            'p-value': p_value,
            'Reject Null Hypothesis at 5%': reject_null
        })
    # Convert the results to a DataFrame
    adf_df = pd.DataFrame(results)
    return adf_df