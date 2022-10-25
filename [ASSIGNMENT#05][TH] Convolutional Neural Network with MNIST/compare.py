import pandas 
import matplotlib.pyplot as plt
import numpy as np
from helper_function import *

df = pandas.read_csv('statistics.csv')
data = np.array(df)
models = data[:, 0]
params = data[:, 1]
times = data[:, 2]
accs = data[:, 3]

plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plot_bar(models, params, color='maroon', x_label="Models", y_label="Params", title="PARAMS")

plt.subplot(1, 3, 2)
plot_bar(models, times, color='blue', x_label="Models", y_label="Training Time (S)", title="TRAINING TIME")

plt.subplot(1, 3, 3)
plot_bar(models, np.array([round(i, 3) for i in accs]), color='green', x_label="Models", y_label="Accuracy (%)", title="ACCURACY")

plt.savefig('statistics.png')
plt.show()