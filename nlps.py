import tensorflow as tf
from tensorflow import keras
import numpy as np 

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# Integer encoded words, each of these integers point to a certain word 
# Given one integer, not nice given the computer, find the mapping given these words 
# Create your own mappings for words 
# print(train_data[0])

# print(data[0])
word_index = keras.datasets.imdb.get_word_index() #what are tuples  


# What is going on right here???
# word_index = {k:(v+3) for k, v in word_index.items()}
# print(word_index)
# word_index['<PAD>'] = 0
# word_index['<START>'] = 1
# word_index['<UNK>'] = 2
# word_index['<UNUSED>'] = 3

# for i in range(5):
