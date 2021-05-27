import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from scipy.stats import gamma, poisson
import epyestim
import epyestim.covid19 as covid19


s = np.array(shape=(1,2))
print(s.shape())