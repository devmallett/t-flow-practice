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

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)



# print("This is 0",len((test_data[0])))
# print("This is one",len((test_data[1])))

# Preprocessing data  decoding testing/ trainig data
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

print("This is 0",len(decode_review((test_data[0]))))
print("This is one",len(decode_review((test_data[1]))))

#Defining Model 
'''

High Level Understanding 
CPU doenst have a good understanding of the differences between the words

Embedding Layer
    0     1     2       3
    Have  a     great   day

    0     1     4       3       
    Have  a     good    day 

    [0, 1, 2, 3] - > integer encoded list 

    [0, 1, 4, 3] - > integer encoded list 

    -All we can tell is 2 is differnt from 4, we know that they have a similiar context 
    -Want to have words that are similiar in context even though words are different 
    -Embedding layer groupd words in a similiar way so they know 
    -Generates word vectors to past to future layers, any dimensional space 




    We know these are different 

     
    -

'''

model = keras.Sequential()
# Embedding Layer 
model.add(keras.layers.Embedding(10000, 16, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None))
#Global Average Pooling 1D - takes what ever dimension our data is in and puts it in a lower dimension 
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid")) #squish everything, what ever our value is between 0 and one 

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) #loss will calculate difference


#Validation Data - can check how well the model is performing based on the qtweaks we make for new data 

x_val = train_data[:10000] #size doesnt matter
x_train = train_data[10000:]

y_val = train_labels[:10000] #size doesnt matter
y_train = train_labels[10000:]


fit_model = model.fit(x_train, y_train, epochs=40, batch_size=512,validation_data=(x_val, y_val), verbose=1)
#Hyper tuning, changing individual parameters / fine tuning 

results = model.evaluate(test_data, test_labels)

print(results)

#Saves model in bianry data 
#
model.save("model.h5")

model = keras.models.load_model("model.h5")






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