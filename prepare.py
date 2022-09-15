import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



def prepare_data(train, test):
    """Make the data ready for ML model
    Returns: (X_train, y_train, X_test, full_pipe)
    """
    # create day, month and year from Date column

    train['Day'] = train['Date'].dt.day
    test['Day'] = train['Date'].dt.day

    train['Month'] = train['Date'].dt.month
    test['Month'] = train['Date'].dt.month

    train['Year'] = train['Date'].dt.year
    test['Year'] = train['Date'].dt.year

    # Drop Date column
    train.drop("Date", axis=1, inplace=True)
    test.drop("Date", axis=1, inplace=True)

    # drop extra column from the training set
    train.drop("#Order",axis=1, inplace=True)
    # set the ID column as index as we are not allowed to use this column
    train.set_index('ID', inplace=True)
    test.set_index('ID', inplace=True)

    # convert int cols that should be categorical
    train['Holiday'] = train['Holiday'].map({0:"No", 1:"Yes"})
    test['Holiday'] = test['Holiday'].map({0:"No", 1:"Yes"})

    train['Day'] = train['Day'].astype(str)
    test['Day'] = test['Day'].astype(str)

    train['Month'] = train['Month'].astype(str)
    test['Month'] = test['Month'].astype(str)

    train['Year'] = train['Year'].astype(str)
    test['Year'] = test['Year'].astype(str)

    # seperate features and target
    X_train = train.drop("Sales",axis=1).copy()
    y_train = train['Sales'].copy()
    X_test = test.copy()

    # select numerical and categorical columns
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    # numerical pipeline
    num_pipe = make_pipeline(SimpleImputer(strategy="mean"))

    # categorical pipeline
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NA"),
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )

    # full pipeline for data preprocessing
    full_pipe = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )
    full_pipe

    return X_train, y_train, X_test, full_pipe

