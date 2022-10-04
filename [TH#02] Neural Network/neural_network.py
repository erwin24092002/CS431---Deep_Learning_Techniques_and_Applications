import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Input, Model
from keras.models import load_model
from matplotlib.colors import ListedColormap



class neural_network():
    def __init__(self):
        return None

    def build(self, in_dim = 1):
        input = Input(in_dim)
        hidden = Dense(5, use_bias=True, activation='sigmoid')(input)
        output = Dense(1, use_bias=True, activation='sigmoid')(hidden)
        self.model = Model(input, output)
        return self.model

    def train(self, x_train, y_train):
        self.model.compile(optimizer="Adam", loss="binary_crossentropy")
        hist = self.model.fit(x_train, y_train, epochs=5000)
        return hist

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

# Step 1: Generate and visualize data
np.random.seed(1907)
tf.random.set_seed(1907)
x = np.random.rand(50, 2)*30
y = []
for i in range(x.shape[0]):
    y.append(1 if (x[i, 0]-15)**2 + (x[i, 1]-15)**2 > 121 else 0 )
y = np.array(y)

# Step 2: Build model 
NNmodel = neural_network()
NNmodel.build(2)
NNmodel.summary()

# Step 3: Train model
hist = NNmodel.train(x, y)
NNmodel.save("MyNeuralNetwork")
NNmodel.load("MyNeuralNetwork")

# Step 4: Test and visualize model
params = NNmodel.get_trained_params()
a = params[0]
bias = params[1]
print('Trained params: ', a, bias)


# Visualize
plt.figure(figsize=(13, 5))

plt.subplot(1, 3, 1)
for i in range(len(y)):
    if y[i]==0: 
        plt.scatter(x[i, 0], x[i, 1], color='red')
    else: 
        plt.scatter(x[i, 0], x[i, 1], color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")

plt.subplot(1, 3, 2)
plt.plot(hist.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Loss")


plt.subplot(1, 3, 3)
colors = ('red','blue')
cmap = ListedColormap(colors[:len(np.unique(y))])

x_min, x_max = min(x[:, 0]) - 3, max(x[:, 0]) + 3
y_min, y_max = min(x[:, 1]) - 3, max(x[:, 1]) + 3
x1, x2 = np.meshgrid(np.arange(x_min, x_max, 0.5),
                        np.arange(y_min, y_max, 0.2))
Z = []
for i in range(x1.shape[0]):
    x_gr = np.concatenate(([x1[i]], [x2[i]]), axis=0)
    Z.append(((NNmodel.predict(x_gr.T)).T)[0])
Z = np.array(Z)
plt.contourf(x1, x2, Z, alpha=0.4, cmap=cmap)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i in range(x.shape[0]):
    if y[i]==0: 
        plt.scatter(x[i, 0], x[i, 1], color='red')
    else: 
        plt.scatter(x[i, 0], x[i, 1], color='blue')
# plt.plot(x, SoftmaxModel.predict(x), color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Model")

plt.savefig("neuralnetwork_result.png")
plt.show()