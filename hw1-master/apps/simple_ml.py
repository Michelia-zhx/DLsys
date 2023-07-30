import struct
import gzip
import numpy as np

import sys

sys.path.append('./python')
import needle as ndl
from needle.ops import exp, summation, log, relu, matmul
from needle.autograd import Tensor

def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filesname, 'rb') as f_img:
        image_content = f_img.read()
        # endianness = [
        #     ('@', 'native, native'),
        #     ('=', 'native, standard'),
        #     ('<', 'little-endian'),
        #     ('>', 'big-endian'),
        #     ('!', 'network'),
        # ] 各种字节顺序和其编码
        (magic_number, n_img, n_row, n_column) = struct.unpack('>'+'iiii', image_content[:16])
        X = np.frombuffer(image_content, dtype=np.ubyte, offset=16)
        X = X.reshape(n_img, n_row * n_column)
        min = X.min()
        max = X.max()   # 这里应该根据全局的最大最小值对所有图像做一样的归一化
        X = X / (max - min)
        X = X.astype(np.float32)
        # print(magic_number, n_img, n_row, n_column)
        # print(X.shape)
    
    with gzip.open(label_filename, 'rb') as f_label:
        label_content = f_label.read()
        (magic_number, n_label) = struct.unpack('>'+'ii', label_content[:8])
        y = np.frombuffer(label_content, dtype=np.ubyte, offset=8)
        y = y.astype(np.uint8)
        # print(label_number, n_label)
        # print(y[:10])
    
    return (X, y)


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    n = Z.shape[0]
    x = summation(exp(Z), axes=(1,))
    y = summation(log(x))
    z = summation(Z * y_one_hot)
    loss = y - z
    return loss / n

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    
    n = X.shape[0]
    n_class = max(y) + 1
    for idx, bat in enumerate(range(0, n, batch)):
        bat_x = X[bat:bat + batch, :]
        bat_y = y[bat:bat + batch]

        img = Tensor(bat_x)
        label = np.zeros((batch, n_class))
        label[range(batch), bat_y] = 1
        label = Tensor(label)

        z = matmul(relu(matmul(img, W1)), W2)
        loss = softmax_loss(z, label)
        
        loss.backward()
        new_W1 = Tensor(W1.numpy() - lr * W1.grad.numpy())
        new_W2 = Tensor(W2.numpy() - lr * W2.grad.numpy())
        W1, W2 = new_W1, new_W2
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
