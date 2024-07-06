# batch gradient descent
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

file_path = "D:/6th semester/artificial intelligence/lab/python labs/neuron.py/insurance_dataset.xlsx"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Try reading the file
    try:
        df = pd.read_excel(file_path)
        # print(df.head())
    except Exception as e:
        print(f"An error occurred: {e}")


from sklearn.model_selection import train_test_split
# splitting data for train and testing
x_train, x_test, y_train, y_test = train_test_split(df[['age','affordibility']],df.have_insurance,test_size=0.2, random_state=25)
print(len(x_train))
# scalling age to be on same scale as affordibility

x_train_scaled = x_train.copy()
x_train_scaled['age'] = x_train_scaled['age'] / 100

x_test_scaled = x_test.copy()
x_test_scaled['age'] = x_test_scaled['age'] / 100

