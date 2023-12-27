import numpy as np

def linear_log_arange(start):
    assert start > 0, "start must be greater than 0"
    numbers = []
    for exponent in range(-start, 0):
        numbers.extend([(i * 0.05) * 10**exponent for i in range(1, 21)])
    return np.array(numbers)