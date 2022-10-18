import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Input, Model, datasets, losses
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns


class neural_network():
    def __init__(self):
        return None

    def build(self, in_dim = 1, out_dim = 1):
        input = Input(in_dim)
        hidden = Dense(8, use_bias=True, activation='relu')(input)
        # hidden = Dense(64, use_bias=True, activation='relu')(hidden)
        # hidden = Dense(32, use_bias=True, activation='relu')(hidden)
        output = Dense(out_dim, use_bias=True, activation='softmax')(hidden)
        self.model = Model(input, output)
        return self.model

    def train(self, x_train, y_train, epochs=500):
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9)
        self.model.compile(optimizer=optimizer, loss=losses.SparseCategoricalCrossentropy())
        hist = self.model.fit(x_train, y_train, epochs=epochs)
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


# Step 1: Load MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five',
               'Six', 'Seven', 'Eight', 'Nine']

x_train = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
x_test = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
x_train = x_train / 255.0 
x_test = x_test / 255.0 

y_train = train_labels
y_test = test_labels


# Step 2: Build model 
NNmodel = neural_network()
NNmodel.build(in_dim=x_train.shape[1], out_dim=len(class_names))
NNmodel.summary()


# Step 3: Train model
hist = NNmodel.train(x_train, y_train, 5)
NNmodel.save("MyNeuralNetwork")
NNmodel.load("MyNeuralNetwork")


# Step 4: Visualize Result
# MINIST Data
plt.figure(figsize=(12,12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig('visual_data.png')
plt.show()

# Result
predictions = NNmodel.predict(x_test)
print(predictions)

def visualize_result(y_test, y_predict, title):
    fig, ax = plt.subplots(figsize=(10,4))
    cm = confusion_matrix(y_test, np.argmax(y_predict, axis=1))
    sns.heatmap(data = cm, cmap="Blues",
            annot=True, fmt=".2f",
            linecolor='white', linewidths=0.5)
    yticks = class_names
    xticks = class_names
    ax.set_yticklabels(yticks, rotation=0)
    ax.set_xticklabels(xticks, rotation=0)
    ax.set_xlabel('Groundtruth', color='red')
    ax.set_ylabel('Predict', color='red')
    ax.set_title(title, color='red')
visualize_result(test_labels, predictions, 'RESULT')
plt.savefig('result.png')
plt.show()

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array),
                                    class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    idx = i*i
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(idx, predictions[idx], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(idx, predictions[idx], test_labels)
plt.tight_layout()
plt.savefig("visual_result.png")
plt.show()
