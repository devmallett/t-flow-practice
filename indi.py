import tensorflow as tf
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt
# # from matplotlib import pyplot
# # you dont want to pass all opf your data into the network when you train it 
# #want to test the data for accuracy
# # if you test your network on data that its already seen, cant be sure that it is the data that we have seen 


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
# plt.imshow(train_images[5])

train_images = train_images/255.0
test_images = test_images/255.0
# print(train_labels[0])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu") , #rectify linear unit 
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=9)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# # python indi.py

# print("test acc", test_acc)
# # 


# Variable Preduiction

prediction = model.predict([test_images])
print( class_names[np.argmax(prediction[0]) ])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[0])])
    plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# x = np.arange(0, 5, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()

