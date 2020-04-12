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
word_index = data.get_word_index() #tuples that has key and mappings  
# print(word_index)

# python nlps.py

# Word Mapping to - what does this do though???
# Finding a way to diplay it so we can look at it
# Breaking up the tuple into key and avlue 
 #PRIOR TO THIS LINE BELOW DATA STILL RETURNS SAME DATA AS LINE 15 
 #THERE ARE SUPPOSED TO BE THREE KEYS FOR SPECIAL INDEXING 

word_index = {k:(v+3) for k, v in word_index.items()} 
print(word_index)

#aLL OF THE WORDS IN TRAINING AND TESTING DATA SET HAVE KEYS AND VLAUES ASSOCIATED WITH THEM 
#CAN ASSIGN YOUR OWN VALUES THAT DEAL WITH PAD, START, UNK, UNUSED 

word_index['<PAD>'] = 0 #MAKING EACH MOVIE REVIEW TO BE THE SAME LENGTH 
word_index['<START>'] = 1
word_index['<UNK>'] = 2 #STANDS FOR UNKNOWN
word_index['<UNUSED>'] = 3

# for i in range(5):

# reverse_word_index = dict([(value, key) for (key, value) in word_index.item()])


