import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Input, Model
from keras.models import load_model

class linear_regression():
    def __init__(self):
        return None

    def build(self, in_dim = 1):
        input = Input(in_dim)
        output = Dense(1, use_bias=True)(input)
        self.model = Model(input, output)
        return self.model

    def train(self, x_train, y_train):
        self.model.compile(optimizer="SGD", loss="mse")
        hist = self.model.fit(x_train, y_train, epochs=1000)
        return hist

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        self.model = load_model(model_path)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def summary(self):
        print("Number of layers: ", len(self.model.layers))
        return self.model.layers[1].get_weights()

    def get_trained_params(self):
        print("Number of layers: ", len(self.model.layers))
        return self.model.layers[1].get_weights()

# Step 1: Generate data
np.random.seed(1)
tf.random.set_seed(1)
x = np.arange(3,7,0.5)
y = -8*x + 13 + 1*np.random.randn(len(x))

# Step 2: Build model 
LinearModel = linear_regression()
LinearModel.build(1)
LinearModel.summary()

# Step 3: Train model
hist = LinearModel.train(x, y)
LinearModel.save("MyLinearModel")
LinearModel.load("MyLinearModel")

# Step 4: Test and visualize model
params = LinearModel.get_trained_params()
a = params[0][0][0]
bias = params[1][0]
print('Trained params: ', a, bias)

# Visualize
plt.figure(figsize=(13, 5))

plt.subplot(1, 3, 1)
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")

plt.subplot(1, 3, 2)
plt.plot(hist.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Loss")

plt.subplot(1, 3, 3)
plt.scatter(x, y)
plt.plot(x, LinearModel.predict(x), color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Model")

plt.savefig("linear_visual.png")
plt.show()
