def plot_history(history, epochs):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), history['loss'], label='train loss')
    plt.plot(np.arange(0, epochs), history['accuracy'], label='train accuracy')
    plt.plot(np.arange(0, epochs), history['val_loss'], label='validation loss')
    plt.plot(np.arange(0, epochs), history['val_accuracy'], label='validation accuracy')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss\Accuracy")
    plt.legend()
    plt.show()
