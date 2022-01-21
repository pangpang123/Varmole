import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

x = pd.read_csv(r'project84.csv')
x = x.set_index('id')
plt.xticks(x['w'])
plt.show()

# frequencies


import numpy
from scipy.signal import find_peaks

test = np.array(x['w'])
test[0:10] = 0
peaks, peak_plateaus = find_peaks(- test, plateau_size = 1)
for i in range(len(peak_plateaus['plateau_sizes'])):
    if peak_plateaus['plateau_sizes'][i] > 1:
        print('a plateau of size %d is found' % peak_plateaus['plateau_sizes'][i])
        print('its left index is %d and right index is %d' % (peak_plateaus['left_edges'][i], peak_plateaus['right_edges'][i]))
