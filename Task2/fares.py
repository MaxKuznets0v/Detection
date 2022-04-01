from Task2.methods import *
import cv2
import matplotlib.pyplot as plt


class FaReS:
    def __init__(self, base, num_classes, col_size=8):
        self.cur_test = -1
        self.col, self.test = build_collection(build_targets(base), col_size)
        self.methods = [Histogram(self.col), Scale(self.col), Gradient(self.col), DFT(self.col), DCT(self.col)]
        self.classes = num_classes

    def pick_next(self):
        self.cur_test += 1
        return self.test[self.cur_test]

    def test_graph(self):
        accuracies = list()
        total = 0
        for i, elem in enumerate(self.test):
            im, guess, *_ = self.predict(elem[0])
            if guess == elem[1]:
                total += 1
            print(f"Test {i + 1}: {total / (i + 1)}")
            accuracies.append(total / (i + 1))
        return accuracies

    def predict(self, image, show=False):
        occ = [list() for i in range(self.classes + 1)]
        features = dict()
        features['Original'] = image
        images = dict()
        images['Original'] = image
        for method in self.methods:
            guess, diff, path = method.predict(image)
            if show:
                if method.__class__.__name__ in ['Scale', 'DFT', 'DCT']:
                    features[method.__class__.__name__] = method.extractor(image, vec=False)
                else:
                    features[method.__class__.__name__] = method.extractor(image)
                images[method.__class__.__name__] = cv2.imread(path)
            occ[guess].append(path)
        fig = None
        if show:
            fig = self.show_res(features, images)

        best = list()
        res = 0
        for i, elem in enumerate(occ):
            if len(elem) > len(best):
                best = elem
                res = i
        return cv2.imread(best[0]), res, len(occ[res]) / len(self.methods), fig

    def eval(self):
        cnt = 0
        for tst in self.test:
            im, guess, *_ = self.predict(tst[0])
            if guess == tst[1]:
                cnt += 1
        return cnt / len(self.test)

    def show_res(self, features, results):
        fig = plt.figure(figsize=(12, 7))
        columns = 6
        rows = 2
        for i, key in enumerate(features):
            sub = fig.add_subplot(rows, columns, i + 1)
            sub.title.set_text(key)
            if key == 'Original':
                sub.set_ylabel('Feature', fontsize=13)
            if key == 'Histogram' or key == 'Gradient':
                plt.plot(features[key])
            else:
                plt.imshow(features[key], cmap='gray', aspect="auto")
            plt.xticks([])
            plt.yticks([])

        for i, key in enumerate(results):
            sub = fig.add_subplot(rows, columns, i + 7)
            if key == 'Original':
                sub.set_ylabel('Result', fontsize=13)
            plt.imshow(results[key], cmap='gray', aspect="auto")
            plt.xticks([])
            plt.yticks([])
        return fig
