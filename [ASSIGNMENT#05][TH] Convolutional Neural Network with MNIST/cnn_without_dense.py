from turtle import color
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras import Input, Model, datasets, losses, layers, models
from keras.models import load_model
import seaborn as sns
import time
import pandas
from helper_function import *


class cnn():
    def __init__(self):
        return None

    def build(self, in_dim = (1, 1), out_dim = 1):
        input = Input(in_dim)

        hidden = layers.Conv2D(8, (3, 3), activation='relu') (input)
        hidden = layers.MaxPooling2D((2, 2)) (hidden)

        hidden = layers.Conv2D(32, (3, 3), activation='relu') (hidden)
        hidden = layers.MaxPooling2D((2, 2)) (hidden)

        hidden = layers.Conv2D(10, (4, 4), activation='softmax') (hidden)
        hidden = layers.MaxPooling2D((2, 2)) (hidden)

        output = layers.Flatten() (hidden)

        self.model = Model(input, output)
        return self.model

    def train(self, x_train, y_train, epochs=500):
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9)
        loss = losses.SparseCategoricalCrossentropy()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        start = time.time()
        self.hist = self.model.fit(x_train, y_train, epochs=epochs)
        end = time.time()
        model.train_t = end - start
        return self.hist

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        self.model = load_model(model_path)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def summary(self):
        self.model.summary()
    
    def get_trained_params(self):
        print("Number of layers: ", len(self.model.layers))
        return self.model.layers[1].get_weights()

    def save_statistic(self, file_name):
        idx = 2
        df = pandas.read_csv(file_name)
        df.at[idx, 'Training Time'] = self.train_t

        train_params = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        non_train_params = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        df.at[idx, 'Params'] = train_params + non_train_params

        df.at[idx, 'Accuracy'] = self.hist.history['accuracy'][-1]
        
        df.to_csv(file_name, index=False)

# Step 1: Load MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five',
               'Six', 'Seven', 'Eight', 'Nine']

x_train = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
x_test = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
x_train = x_train / 255.0 
x_test = x_test / 255.0 

y_train = train_labels
y_test = test_labels


# Step 2: Build model 
model = cnn()
model.build(in_dim=x_train[0].shape, out_dim=len(class_names))
model.summary()


# Step 3: Train model
hist = model.train(x_train, y_train, 3)
model.save("CNN WITHOUT DENSE")
model.load("CNN WITHOUT DENSE")


# Step 4: Visualize Result
PATH = "visual/cnn without dense/"
# MINIST Data
visualize_data(train_images, train_labels, class_names)
plt.savefig('data.png')
plt.show()

# Training
visualize_training(hist)
plt.savefig(PATH + 'training.png')
plt.show()

# Result
predictions = model.predict(x_test)
visualize_result(test_labels, predictions, 'CONFUSION MATRIX', class_names)
plt.savefig(PATH + 'confusion_matrix.png')
plt.show()

# Example
visualize_example(predictions, test_labels, test_images, class_names)
plt.savefig(PATH + "result.png")
plt.show()

# save statistic
model.save_statistic('statistics.csv')
