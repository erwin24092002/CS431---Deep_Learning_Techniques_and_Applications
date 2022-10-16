import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Input, Model
from keras.models import load_model


class logistic_regression():
    def __init__(self):
        return None

    def build(self, in_dim = 1):
        input = Input(in_dim)
        output = Dense(1, use_bias=True, activation='sigmoid')(input)
        self.model = Model(input, output)
        return self.model

    def train(self, x_train, y_train):
        self.model.compile(optimizer="SGD", loss="binary_crossentropy")
        hist = self.model.fit(x_train, y_train, epochs=1000)
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
np.random.seed(1)
tf.random.set_seed(1)
x = np.random.rand(30, 2)*30
y = []
for i in range(x.shape[0]):
    y.append(1 if x[i, 0] - 0.95*x[i, 1]+ 2 > 0 else 0 )
y = np.array(y)

# Step 2: Build model 
LogisticModel = logistic_regression()
LogisticModel.build(2)
LogisticModel.summary()

# Step 3: Train model
hist = LogisticModel.train(x, y)
LogisticModel.save("MyLogisticModel")
LogisticModel.load("MyLogisticModel")

# Step 4: Test and visualize model
params = LogisticModel.get_trained_params()
a = params[0]
bias = params[1][0]
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
for i in range(len(y)):
    if y[i]==0: 
        plt.scatter(x[i, 0], x[i, 1], color='red')
    else: 
        plt.scatter(x[i, 0], x[i, 1], color='blue')
c = -bias/a[1]
m = -a[0]/a[1]
xmin, xmax = -30, 30
ymin, ymax = m*xmin+c, m*xmax+c
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Model")

plt.savefig("logistic_visual.png")
plt.show()