import numpy as np


class TemplateMatching:
    def __init__(self, img):
        self.image = self.preload(img)

    @staticmethod
    def dist(template, current):
        err = 0
        for i in range(template.shape[0]):
            for j in range(template.shape[1]):
                err += (int(template[i][j]) - int(current[i][j]))**2
        return (err / template.size)**(1/2)

    @staticmethod
    def preload(img):
        return np.array(img.convert('L'))

    def find_one(self, template, delta, threshold):
        sample = self.preload(template)
        best_score = 100000
        best_point = None

        for y in range(0, self.image.shape[0] - sample.shape[0], delta):
            for x in range(0, self.image.shape[1] - sample.shape[1], delta):
                cur_dist = self.dist(sample, self.image[y:y + sample.shape[0], x:x + sample.shape[1]])
                if cur_dist < threshold and cur_dist < best_score:
                    best_score = cur_dist
                    best_point = (x, y)
                    print("New best:", best_score, best_point)
        if best_point is not None:
            return [best_point, (best_point[0] + sample.shape[1], best_point[1] + sample.shape[0])]
        return best_point

    def find_all(self, template, delta, threshold):
        objects = list()
        obj = self.find_one(template, delta, threshold)
        while obj is not None:
            objects.append(obj)
            self.image[obj[0][1]:obj[1][1], obj[0][0]:obj[1][0]] = 255
            obj = self.find_one(template, delta, threshold)
        print(f"Found {len(objects)} objects")
        return objects
