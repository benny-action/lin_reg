from math import sqrt
import numpy as np

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for y, y_hat in zip(actual,predicted):
        prediction_error = y - y_hat
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

def compute_coefficient(x, y):
    n = len(x)
