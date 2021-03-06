import cv2
import os
from abc import ABCMeta, abstractmethod
import numpy as np
import random
from scipy.fftpack import dct


class Method(metaclass=ABCMeta):
    def __init__(self, collection):
        self.col = self.extract(collection)

    def extract(self, col):
        return [[self.extractor(elem[0]), elem[1], elem[2]] for elem in col]

    @abstractmethod
    def extractor(self, image):
        pass

    def predict(self, image):
        best_diff = None
        guess = None
        best_im = None
        cur_im = self.extractor(image)
        for im_cat in self.col:
            cat_im = im_cat[0]
            diff = self.compare(cur_im, cat_im)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                guess = im_cat[1]
                best_im = im_cat[2]
        return guess, best_diff, best_im

    def compare(self, one, another):
        return (np.square(np.array(one) - np.array(another))).mean(axis=None)


class Histogram(Method):
    def __init__(self, collection, bin_size=20):
        self.bin_size = bin_size
        super().__init__(collection)

    def extractor(self, image):
        hist = cv2.calcHist([image], [0], None, [self.bin_size], (0, 256), accumulate=False)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def compare(self, one, another):
        return cv2.compareHist(one, another, 3)


class DFT(Method):
    def __init__(self, collection, p=9):
        self.p = p
        super().__init__(collection)

    def extractor(self, image, vec=True):
        dft = np.abs(np.fft.fft2(image))
        if vec:
            vector = np.zeros(int(self.p * (self.p + 1) / 2))
            cur = -1
            for i in range(self.p):
                for j in range(i, -1, -1):
                    cur += 1
                    vector[cur] = dft[i - j, j]
            return vector
        return dft[:self.p, :self.p]


class DCT(Method):
    def __init__(self, collection, p=15):
        self.p = p
        super().__init__(collection)

    def extractor(self, image, vec=True):
        dct_res = np.abs(dct(dct(image.T).T))
        if vec:
            vector = np.zeros(int(self.p * (self.p + 1) / 2))
            cur = -1
            for i in range(self.p):
                for j in range(i, -1, -1):
                    cur += 1
                    vector[cur] = dct_res[i - j, j]
            return vector
        return dct_res[:self.p, :self.p]


class Scale(Method):
    def __init__(self, collection, size=(20, 18)):
        self.size = size
        super().__init__(collection)

    def extractor(self, image, vec=True):
        img = cv2.resize(image, self.size)
        if vec:
            img = img.ravel()
        return img


class Gradient(Method):
    def __init__(self, collection, width=1, step=1):
        self.width = width
        self.step = step
        super().__init__(collection)

    def extractor(self, image):
        res = list()
        for s in range(0, image.shape[0] - 2 * self.width, self.step):
            upper = image[s:s+self.width, :]
            lower = image[s + self.width:s + 2 * self.width, :]
            res.append(np.sum(np.power(upper-lower, 2))**(1/2))
        return res


def iterate_folder(path):
    res = list()
    for subdir, dirs, files in os.walk(path):
        for file in files:
            res.append(os.path.join(subdir, file))
    return res


def build_targets(path):
    tar = list()
    for file in iterate_folder(path):
        tar.append([cv2.imread(file, cv2.IMREAD_GRAYSCALE), int(file.split('_')[1].split('.')[0]), file])
    return tar


def build_collection(full, num=5):
    classes = {elem[1]: list() for elem in full}
    for elem in full:
        classes[elem[1]].append([elem[0], elem[2]])
    collection = list()
    for key in classes:
        for im, path in random.sample(classes[key], num):
            collection.append([im, key, path])

    test = list()
    for elem in full:
        add = True
        for cl in collection:
            if (elem[0] == cl[0]).all():
                add = False
        if add:
            test.append(elem)
    return collection, test
