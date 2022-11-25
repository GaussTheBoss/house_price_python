"""
This file contains imports and helper functions needed by main script: house_price.py
"""

import logging
import pickle
from typing import List

import pandas
import numpy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def process(
    input_data: pandas.DataFrame,
    train_encoded_columns: List[str],
    apply_scaler: bool,
    standard_scaler: StandardScaler = None,
) -> pandas.DataFrame:
    """
    A function to apply imputations, feature engineering, scaling to input data.

    Args:
        input_data (pandas.DataFrame): Data to be processed.
        train_encoded_columns (List[str]): Columns produced and encoded by the training step.
        apply_scaler (bool): Flag to indicate whether to apply a pre-configured standard scaler.
        standard_scaler (sklearn.preprocessing._data.StandardScaler): Pre-trained standard scaler.
            Only used if apply_scaler=True.

    Returns:
        (pandas.DataFrame): Processed DataFrame.
    """

    # Dictionaries to convert values in certain columns
    generic = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    garage_finish = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}

    # Imputations
    logger.info("Imputing input dataset")
    input_data["GarageYrBlt"] = input_data["GarageYrBlt"].fillna(
        input_data["YearBuilt"]
    )
    for col in ["GarageFinish", "BsmtQual", "FireplaceQu"]:
        input_data[col] = input_data[col].fillna("None")
    # The rest of NaNs will be filled with 0s - end model only uses numerical features
    for col in input_data.columns:
        input_data[col] = input_data[col].fillna(0)

    generic_columns = [
        "ExterQual",
        "BsmtQual",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
    ]
    # Converting categorical values from certain features into numerical
    for col in generic_columns:
        input_data[col] = input_data[col].map(generic)

    input_data["GarageFinish"] = input_data["GarageFinish"].map(garage_finish)

    # Feature engineering
    input_data["eHasFireplace"] = input_data["Fireplaces"] > 0
    input_data["eTotalSF"] = (
        input_data["TotalBsmtSF"] + input_data["1stFlrSF"] + input_data["2ndFlrSF"]
    )
    input_data["eTotalBathrooms"] = (
        input_data["FullBath"]
        + (0.5 * input_data["HalfBath"])
        + input_data["BsmtFullBath"]
        + (0.5 * input_data["BsmtHalfBath"])
    )
    input_data["eOverallQual_TotalSF"] = (
        input_data["OverallQual"] * input_data["eTotalSF"]
    )

    input_data["Foundation_PConc"] = input_data["Foundation"] == "PConc"

    # Limiting features to just the ones the model needs
    logger.info("Selecting columns that model is expecting")
    input_data = input_data[train_encoded_columns]

    if apply_scaler:
        # Scale inputs
        logger.info("Scaling data with pickled standard scaler")
        input_data = pandas.DataFrame(
            standard_scaler.transform(input_data.values),
            index=input_data.index,
            columns=input_data.columns,
        )

    return input_data
