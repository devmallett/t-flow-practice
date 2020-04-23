import tensorflow as tf
    
import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 
import os 
import pandas as pd

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['facecolor'] = 'darkolivegreen'

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)

df.head()

# print(df)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labeles = []

    start_index = start_index + history_size

if end_index is None:
    end_index= len(dataset) - target_size

for i in range(start_index, end_index):
    indicies = range(i-history_size, i)
    data.append(np.reshape(dataset[indicies], (history_size, 1)))
    labels.append(dataset[i+target_size])
    return np.array(Data), np.array(labels)

TRAIN_SPLIT = 300000
tf.random.set_seed(13)


# univariate
uni_data = df['T (deg C)']
uni_data.index = df[' Date Time ']
uni_data.head()

print(uni_data)





# python ts-prac.py
