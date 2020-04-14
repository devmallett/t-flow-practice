import tensorflow as tf
from tensorflow import keras 
import numpy as np 


data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# Integer encoded runs, makes our model easy to classify, added integers 
# that represent the review 
# have to find the mappings, have to create your own mapping and dictionaries 

print(train_data[0])

word_index = data.get_word_index()


# 3 keys that will be special characters for word mapping
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


# GET VALUES THAT ARE NOT VALID, CAN STORE THEM INTO THE DICTIONARY 

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

# print(reverse_word_index)

def decode_reviewed(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])
print(decode_reviewed(test_data[0]))


# python mov.py