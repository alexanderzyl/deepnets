
def plot_history(history, epochs):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), history['loss'], label='loss')
    plt.plot(np.arange(0, epochs), history['accuracy'], label='accuracy')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss\Accuracy")
    plt.legend()
    plt.show()