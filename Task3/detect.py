from Task3.methods import *


class StyleDetector:
    def __init__(self, base, col_size):
        self.col, self.test = build_collection(base, col_size)
        self.methods = [SIFT(self.col), LocalBinaryPattern(self.col), ColorHistogram(self.col)]

    def get_features(self, image):
        return {method.__class__.__name__: method.extractor(image) for method in self.methods}

    def predict(self, image):
        votes = dict()
        detected = list()
        for method in self.methods:
            guess, _, path = method.predict(image)
            print(f"Method {method.__class__.__name__} is voting for {guess}")
            try:
                votes[guess] += 1
            except:
                votes[guess] = 1
            detected.append([guess, cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)])
        return max(votes, key=votes.get), detected

    def eval(self):
        acc = 0
        for test in self.test:
            guess = self.predict(test[0])
            print("Correct:", test[1])
            if guess == test[1]:
                acc += 1
        return acc / len(self.test)
