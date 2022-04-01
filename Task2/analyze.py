from Task2.methods import build_collection, iterate_folder
import matplotlib.pyplot as plt
import cv2


def test_method(full, method):
    accuracies = list()
    for col_size in range(1, 10):
        collection, test = build_collection(full, col_size)
        model = method(collection)
        count = 0
        for elem in test:
            guess, *_ = model.predict(elem[0])
            if guess == elem[1]:
                count += 1
        cur_acc = count / len(test)
        accuracies.append(cur_acc)
        print(f"Current accuracy on collection size {col_size}= {cur_acc}")
    plt.plot((1, 10), accuracies, label=method.__name__)


def assign_param(collection, test, method):
    accuracies = list()
    specter = range(1, 20, 3)
    for p in specter:
        if method.__name__ == 'Scale':
            predictor = method(collection, (2 + 2*p, 1 + 2*p))
        else:
            predictor = method(collection, p)
        count = 0
        for elem in test:
            guess, *_ = predictor.predict(elem[0])
            if guess == elem[1]:
                count += 1
        print(f"Param: {p} accuracy = {count / len(test)}")
        accuracies.append(count / len(test))
    plt.plot(specter, accuracies, label=method.__name__)


def validate(sys, size):
    accuracies = list()
    for i in range(1, size):
        pred = sys('orig', 41, i)
        accuracies.append(pred.eval())
        print(f"Size {i} accuracy: {accuracies[i-1]}")
    return accuracies


def test_mask(pred, path, flag):
    accuracy = 0
    files = iterate_folder(path)
    size = 0
    for elem in files:
        if elem.split('\\')[1][0] == flag:
            size += 1
            im, guess, *_ = pred.predict(cv2.imread(elem, cv2.IMREAD_GRAYSCALE))
            if guess == int(elem.split('_')[1].split('.')[0]):
                accuracy += 1

    return accuracy / size
