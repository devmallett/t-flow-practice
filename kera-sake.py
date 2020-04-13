import tensorflow as tf 
from tensorflow import keras
import numpy as np

# from keras.models import Sequential
# from keras.layers import Dense, Activation


# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# data = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000,1))

# one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# model = keras.Sequential([
#     keras.layers.Dense(32, input_shape=(100,)),
#     keras.layers.Activation('relu'),
#     keras.layers.Dense(100),
#     keras.layers.Activation('softmax'),
# ])


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(data, labels, epochs=10, batch_size=32)



# prediction = model.predict([one_hot_labels])
# vals = np.random.random((1000, 100))
vals = np.random.randint(2, size=10)
print(vals)


# python kera-sake.py

# data = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = data.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
# # plt.imshow(train_images[5])

# train_images = train_images/255.0
# test_images = test_images/255.0
# print(train_labels[0])



# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128, activation="relu") , #rectify linear unit 
#     keras.layers.Dense(10, activation="softmax")
# ])

# model.compile(optimizer='adam', 
# loss='sparse_categorical_crossentropy', 
# metrics=['accuracy'])






