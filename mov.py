import tensorflow as tf
from tensorflow import keras 
import numpy as np 
import random


'''
Supervised Learning - its called that because we are supervising what our model is able to see, so it can then identify
later subjects? 
So here is the overall flow of training data 

1. first you have to load botth data sets
    this can be traiing data and testing data 
    
2. next you have to take your training data and get it into a format that you can read 
    this is most likely some splitting, slicing, reversing of the array 
    could also be a time when you have to create an array or a dictionary, all depends 
    if your data does not have a key value system then you have to develop it 
        tensor flow has this own 
        
3. once you have the data ready to be trained, then you have to model the network you want 
    the input layer ends up being hwo ever many points of data you are working with 
    hidden layer is typically around 10~12% of what the input layer is 

4. you may have to run the training simulation more than once to reduce your trainig error 


Still have to dive into Reinforcement Learning and what goes into that 



'''


data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# Integer encoded runs, makes our model easy to classify, added integers 
# that represent the review 
# have to find the mappings, have to create your own mapping and dictionaries 

# print(train_data[3])

word_index = data.get_word_index() #is not an array


# 3 keys that will be special characters for word mapping
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0 #makes each review the same length - adds pad to make the length 200
word_index["<START>"] = 1 
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# decoding testing/ trainig data
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

print("This is 0",len(decode_review((test_data[0]))))
print("This is one",len(decode_review((test_data[1]))))

# print(word_index)

# So if you want to find a random number 
# yummy = random.choice(list(word_index.keys()))

# print(yummy)


# GET VALUES THAT ARE NOT VALID, CAN STORE THEM INTO THE DICTIONARY 
# reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])




# print(word_index)
# checker = len(reverse_word_index) #88,585 items in this tuple
# print(checker)

# def decode_reviewed(text):
#     # takes text in - which by this point ahve been converted from the number value to the text 
#     # one the text is in, you have to reverse 
#     return " (SPACE) ".join([reverse_word_index.get(i, "?") for i in text])
#     # RETURN " ".join([reverse_word_index.get(i, "?") for i in text])

# print(decode_reviewed(train_data[3]))


# python mov.py