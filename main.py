import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import warnings
warnings.filterwarnings(action='ignore')
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
# print(data.head())
print(data.shape)
print(data.isnull().sum())
print("fjkf")
