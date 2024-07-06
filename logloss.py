import numpy as np

epsilon = 1e-15


def log_loss(y_true, y_predicted):
    y_predict_new = [max(i, epsilon) for i in y_predicted]
    y_predict_new = [min(i, 1 - epsilon) for i in y_predict_new]
    y_predict_new = np.array(y_predict_new)
    return -np.mean(y_true*np.log(y_predict_new)+(1-y_true)*np.log(1-y_predict_new))
