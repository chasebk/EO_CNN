import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pandas import read_csv, DataFrame
import numpy as np

## https://matplotlib.org/api/markers_api.html

## Paras

x_label = "Timestamp (5 minutes)"
#y_label = "CPU Usage"
y_label = "Value"
#title = 'Univariate Neural Network'
title = 'Travel-time dataset'

read_filepath = "TravelTime_451.csv"
new_filepath = "travel_time.csv"
write_filepath = "TravelTime_451.pdf"

colnames = ['timestamp', 'value']
results_df = read_csv(read_filepath)

real = results_df['value'].values

x = np.arange(len(real))

# plt.plot(x, real[point_start:point_start + point_number],  marker='o', label='True')
# plt.plot(x, ann[point_start:point_start + point_number],  marker='s', label='ANN')
# plt.plot(x, mlnn[point_start:point_start + point_number],  marker='*', label='MLNN')
# plt.plot(x, flnn[point_start:point_start + point_number],  marker=lines.CARETDOWN, label='FLNN')
# plt.plot(x, flgann[point_start:point_start + point_number],  marker='x', label='FL-GANN')
# plt.plot(x, flbfonn[point_start:point_start + point_number],  marker='+', label='FL-BFONN')

real = np.reshape(real, (-1, 1))

df = DataFrame(real, columns=['value'])
df.to_csv(new_filepath, index=True)

plt.plot(x, real)

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.legend()
plt.savefig(write_filepath, bbox_inches='tight')
plt.show()
