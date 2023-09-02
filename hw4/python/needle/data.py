import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd
import struct
import gzip

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        h, w, c = img.shape
        pad = np.zeros((h + 2*self.padding, w + 2*self.padding, c))
        pad[self.padding:-self.padding, self.padding:-self.padding, :] = img
        x = self.padding + shift_x
        y = self.padding + shift_y
        crop = pad[x:x+h, y:y+w, :]
        return crop


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )
        else:
            arr = np.arange(len(dataset))
            np.random.shuffle(arr)
            self.ordering = np.array_split(arr, range(batch_size, len(dataset), batch_size))
        self.idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= len(self.ordering):
            self.idx = -1
            if self.shuffle:
                arr = np.arange(len(self.dataset))
                np.random.shuffle(arr)
                self.ordering = np.array_split(arr, range(self.batch_size, len(self.dataset), self.batch_size))
            raise StopIteration()
        samples = self.dataset[self.ordering[self.idx]]
        return [Tensor(x) for x in samples]


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        with gzip.open(image_filename) as f_img:
            magic_number, n_img, n_row, n_column = struct.unpack(">IIII", f_img.read(16))
            self.images = np.frombuffer(f_img.read(),
                                        dtype=np.uint8).reshape(-1, 784)

        with gzip.open(label_filename) as f_label:
            magic_number, n_label = struct.unpack(">II", f_label.read(8))
            self.labels = np.frombuffer(f_label.read(), dtype=np.uint8)
        
        self.images = np.float32(self.images) / 255.

    def __getitem__(self, index) -> object:
        if isinstance(index, (Iterable, slice)):
            img = [i.reshape((28, 28, 1)) for i in self.images[index]]
        else:
            img = [self.images[index].reshape((28, 28, 1))]

        if self.transforms:
            for tsf in self.transforms:
                img = [tsf(x) for x in img]

        return [np.stack(img), self.labels[index]]

    def __len__(self) -> int:
        return len(self.images)


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        if train:
            data_batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            data_batch_files = ['test_batch']
        X = []
        Y = []
        for file_name in data_batch_files:
            with open(os.path.join(base_folder, file_name), 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                X.append(data_dict[b'data'])
                Y.append(data_dict[b'labels'])
        X = np.concatenate(X, axis=0)
        X = X / 225.
        X = X.reshape((-1, 3, 32, 32))
        Y = np.concatenate(Y, axis = None)
        self.X = X
        self.Y = Y
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        if self.transforms:
            image = np.array([self.apply_transforms(img) for img in self.X[index]])
        else:
            image = self.X[index]
        label = self.Y[index]
        return image, label

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return len(self.Y)


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.words = set()

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if word not in self.words:
            self.words.add(word)
            uid = len(self.idx2word)
            self.word2idx[word] = uid
            self.idx2word.append(word)
        else:
            uid = self.idx2word.index(word)
        return uid

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.words)



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ids = []
        eos_id = self.dictionary.add_word("<eos>")
        def tokenize_one_line(line):
            words = line.split()
            for word in words:
                ids.append(self.dictionary.add_word(word))
            ids.append(eos_id)
        with open(path, "r") as f:
            if max_lines:
                for _ in range(max_lines):
                    tokenize_one_line(f.readline())
            else:
                for line in f:
                    tokenize_one_line(line)
        return ids


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    nbatch = len(data) // batch_size
    data = np.array(data[:nbatch * batch_size]).reshape((nbatch, batch_size))
    return data


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    tot_seqlen = batches.shape[0]
    assert i < tot_seqlen - 1
    if i + bptt + 1 > tot_seqlen:
        X = batches[i : -1, :]
        y = batches[i+1 : , :].flatten()
    else:
        X = batches[i : i+bptt, :]
        y = batches[i+1 : i+bptt+1, :].flatten()
    return Tensor(X, device=device, dtype=dtype), Tensor(y, device=device, dtype=dtype)