import os
import pandas as pd
import numpy as np
from prepare import prepare_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_log_error

# project paths
data_path = os.path.join(os.getcwd(),'data')

os.makedirs(data_path, exist_ok=True)

# function for reading data
def read_data(filename, date_cols=None, file_path=data_path):
    csv_path = os.path.join(file_path, filename)
    return pd.read_csv(csv_path, parse_dates=date_cols)

# function for saving data as csv file
def save_dataframe(df, filename, file_path=data_path):
    path = os.path.join(file_path, filename)
    df.to_csv(path, index=False)


# read the data
train = read_data("TRAIN.CSV", date_cols=["Date"])
test = read_data("TEST_FINAL.csv", date_cols=["Date"])
submission = read_data("SAMPLE.csv")

# prepare data for machine learning
X_train, y_train, X_test, full_pipe = prepare_data(train, test)

# initiate a random forest model
rf = make_pipeline(full_pipe, RandomForestRegressor(random_state=42, 
                                                    max_depth=30,
                                                    min_samples_leaf= 8, 
                                                    min_samples_split= 6,
                                                    max_features="sqrt",
                                                    n_jobs=-1))
# train the model
rf.fit(X_train, y_train)
# make predictions
y_pred = rf.predict(X_test)

# make submission
submission['Sales'] = y_pred
save_dataframe(submission,"rf_model_final.csv")


