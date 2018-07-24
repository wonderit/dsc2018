# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

train = pd.read_csv("../data/speeddating_train.csv")
test = pd.read_csv("../data/speeddating_test.csv")

train.head()
