import numpy as np

import matplotlib.pyplot as plt

from silly_dl.datasets import load_mnist_one_hot
from silly_dl import layers, network

def main():
    # Load the MNIST dataset
    x_train, labels_train, x_valid, labels_valid = load_mnist_one_hot()
    N_train = x_train.shape[0]

    # Define the network
    nn = network.NeuralNetworkTraining([layers.LinearLayer(784, 40), 
                                        layers.ReLULayer(), 
                                        layers.LinearLayer(40, 10), 
                                        layers.SoftmaxCrossEntropyLayer()]) 

    N_epoch = 20
    batch_size    = 8
    learning_rate = 0.01

    # plotting stuff
    train_losses = []
    val_losses = []
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    train_losses_plot, = ax.plot([], [], label='Train Loss')
    val_losses_plot, = ax.plot([], [], label='Validation Loss')
    ax.legend()
    plt.show()


    Loss_valid = nn.forward(x_valid, labels_valid)
    print(f"Initial loss valid = {Loss_valid}")

    for epoch in range(N_epoch):
        indices = np.random.permutation(N_train)

        losses = []
        for step in range(N_train // batch_size):
            x =      x_train[indices[step*batch_size:(step+1)*batch_size],:]
            l = labels_train[indices[step*batch_size:(step+1)*batch_size],:]

            losses.append(nn.forward(x,l))
            nn.backward()
            nn.update_params(learning_rate)

        print(f"Epoch {epoch} loss train = {np.mean(losses)}")
        Loss_valid = nn.forward(x_valid, labels_valid)
        print(f"Epoch {epoch} loss valid = {Loss_valid}")

        # Update the plot
        train_losses.append(np.mean(losses))
        val_losses.append(Loss_valid)
        train_losses_plot.set_data(range(1, epoch+2), train_losses)
        val_losses_plot.set_data(range(1, epoch+2), val_losses)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    print(f"Training completed")

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()