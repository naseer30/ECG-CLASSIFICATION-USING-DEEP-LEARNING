import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_loss(history):
    plt.figure(figsize=(4, 4))  # Set the figure size

    # Plot training & validation accuracy values
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(4, 4))  # Set the figure size

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Load the training history object
history = np.load('train_history_test3_epoch_15.npy', allow_pickle='TRUE').item()

# Plot accuracy and loss
plot_accuracy_loss(history)
