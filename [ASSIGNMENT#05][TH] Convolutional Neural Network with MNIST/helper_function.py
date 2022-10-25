import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

def visualize_result(y_test, y_predict, title, class_names):
    fig, ax = plt.subplots(figsize=(12,4))
    cm = tf.math.confusion_matrix(labels=y_test, predictions=np.argmax(y_predict, axis=1))
    sns.heatmap(data = cm, cmap="Blues",
            annot=True, fmt=".2f",
            linecolor='white', linewidths=0.5)
    yticks = class_names
    xticks = class_names
    ax.set_yticklabels(yticks, rotation=0)
    ax.set_xticklabels(xticks, rotation=0)
    ax.set_xlabel('PREDICT', color='red')
    ax.set_ylabel('GROUNDTRUTH', color='red', rotation=0)
    ax.set_title(title, color='red')

def plot_image(i, predictions_array, true_label, img, class_names):
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
    plt.ylim([0, max(predictions_array)+1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

def visualize_example(predictions, test_labels, test_images, class_names):
    num_rows = 3
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        idx = i*i
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(idx, predictions[idx], test_labels, test_images, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(idx, predictions[idx], test_labels)
    plt.tight_layout()

def visualize_training(hist):
    plt.plot(hist.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0., max(hist.history['loss'])+1])
    plt.title("TRAINING", color='red')

def visualize_data(images, labels, class_names):
    plt.figure(figsize=(6, 7))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])

def addlabels(x, y, color):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center', fontsize=6, bbox = dict(facecolor = color, alpha =.6))

def plot_bar(x_data, y_data, color, x_label, y_label, title):
    plt.bar(x_data, y_data, color = color, width = 0.6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    addlabels(x_data, y_data, color)
    plt.title(title)