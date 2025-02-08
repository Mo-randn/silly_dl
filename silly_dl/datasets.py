import os
import urllib.request
import gzip
import numpy as np

# Parse images
def parse_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = f.read()
    magic_number, num_images, rows, cols = np.frombuffer(data[:16], dtype=np.uint32).byteswap()
    images = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# Parse labels
def parse_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = f.read()
    magic_number, num_labels = np.frombuffer(data[:8], dtype=np.uint32).byteswap()
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

def preprocess_images(images):
    N = images.shape[0]
    X = images.reshape(N, -1) / 255.0  # flatten image into N x e.g. 784 array (784 for 28x28 image)
    return X

def download_mnist():
    """
    Download the MNIST dataset from the internet if not already done
    """

    # Define the base URL for the dataset
    base_url = "https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/"

    # Define the directory to store the dataset
    data_dir = "data/mnist"

    # Define the file paths
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    # Create the directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download files if they are not already present
    for key, filename in files.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            url = base_url + filename
            urllib.request.urlretrieve(url, file_path)

    # Return the file locations (local path + filename)
    file_locations = {key: os.path.join(data_dir, filename) for key, filename in files.items()}
    return file_locations

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def load_mnist_one_hot():
    mnist_files = download_mnist()

    # Load the dataset only containing 0s and 1s
    images = parse_images(mnist_files["train_images"])
    labels = parse_labels(mnist_files["train_labels"])

    x_train      = preprocess_images(images)
    labels_train = one_hot_encode(labels, 10)

    # Load the dataset only containing 0s and 1s
    images = parse_images(mnist_files["test_images"])
    labels = parse_labels(mnist_files["test_labels"])
    x_valid      = preprocess_images(images)
    labels_valid = one_hot_encode(labels, 10)

    return x_train, labels_train, x_valid, labels_valid
