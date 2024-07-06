import numpy as np
from activation_fuc import sigmoid_numpy
from logloss import log_loss
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


file_path = "D:/6th semester/artificial intelligence/lab/python labs/neuron.py/insurance_dataset.xlsx"
from sklearn.model_selection import train_test_split
# Check if the file exists


def gradient_descent(age, affordability, y_true, epochs):
    # weights and bias
    # w1, w2, bias
    w1 = w2 = 1
    bias = 0
    learning_rate = 0.5
    n = len(age)

    for i in range(epochs):
        weighted_sum = w1*age + w2*affordability + bias
        y_predicted = sigmoid_numpy(weighted_sum)

        loss = log_loss(y_true, y_predicted)
        # derivative
        w1d = (1/n)*(np.dot(np.transpose(age), (y_predicted-y_true)))
        w2d = (1 / n) * (np.dot(np.transpose(affordability), (y_predicted - y_true)))
        biasd = np.mean(y_predicted-y_true)

        w1 = w1 - learning_rate * w1d
        w2 = w2 - learning_rate * w2d
        bias = bias - learning_rate * biasd

        print(f'Epoch:{i}, w1:{w1}, w2{w2}, bias:{bias}, loss:{loss}')

    return w1, w2, bias


if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Try reading the file
    try:
        df = pd.read_excel(file_path)

        # splitting data for train and testing
        x_train, x_test, y_train, y_test = train_test_split(
            df[['age', 'affordibility']], df.have_insurance,test_size=0.2, random_state=25)
        # print(df.head())
        print(len(x_train))
        # scalling age to be on same scale as affordibility

        x_train_scaled = x_train.copy()
        x_train_scaled['age'] = x_train_scaled['age'] / 100

        x_test_scaled = x_test.copy()
        x_test_scaled['age'] = x_test_scaled['age'] / 100

        print(gradient_descent(x_train_scaled['age'], x_train_scaled['affordibility'], y_train, 1000))

    except Exception as e:
        print(f"An error occurred: {e}")


