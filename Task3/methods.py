from Task2.methods import Method, iterate_folder
import numpy as np
import cv2
from skimage import feature


class ColorHistogram(Method):
    def __init__(self, collection, bin_size=30):
        self.bin_size = bin_size
        super().__init__(collection)

    def extractor(self, image):
        res = list()
        for i, color in enumerate(('b', 'g', 'r')):
            hist = cv2.calcHist([image], [i], None, [self.bin_size], [0, 256], accumulate=False)
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            res.append(hist)
        return res


class SIFT(Method):
    def __init__(self, collection):
        self.sift = cv2.xfeatures2d.SIFT_create()
        super().__init__(collection)

    def extractor(self, image):
        image = cv2.resize(image, (200, 200))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = self.sift.detect(img, None)
        return keypoints

    def compare(self, one, another):
        one = np.array([k.pt for k in one])
        another = np.array([k.pt for k in another])
        if one.shape[0] < another.shape[0]:
            one, another = another, one
        res = one.copy()
        res[:another.shape[0], :another.shape[1]] -= another
        return np.sum(np.abs(res))


class LocalBinaryPattern(Method):
    def __init__(self, collection, size=(300, 300), r=9, n=18):
        self.size = size
        self.r = r
        self.n = n
        super().__init__(collection)

    def extractor(self, image):
        image = cv2.resize(image, self.size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = self.r
        n_points = self.n
        target = feature.local_binary_pattern(image, n_points, radius)
        return target


def build_collection(path, size=8):
    train = list()
    test = list()
    for elem in iterate_folder(path):
        target, num = elem.split('\\')[1:3]
        num = int(num.split('.')[0])
        if num <= size:
            train.append([cv2.imread(elem), target, elem])
        else:
            test.append([cv2.imread(elem), target, elem])
    return train, test



