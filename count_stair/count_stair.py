import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from math import atan


class Counter:
    def __init__(self, img):
        self.init(img)

    def init(self, img):
        self.img = np.asarray(img)
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #self.gray = cv2.fastNlMeansDenoising(self.gray, 10, 10, 7, 21)
        cv2.GaussianBlur(self.gray, (3, 3), 0, 0)
        self.edges = cv2.Canny(self.gray, 80, 240, 3)
        self.w = img.shape[0]
        self.h = img.shape[1]
        self.lines = cv2.HoughLinesP(self.edges, 1, np.pi / 180, 30, 40, 15)
        self.peaks = None
        self.descents = None

    def rotation_img(self):
        def cal_d(line):
            delta_x = line[2] - line[0]
            delta_y = line[3] - line[1]
            if delta_x == 0:
                delta_x = 1e-10
            return delta_y / delta_x

        descents_of_lines = np.array([cal_d(x[0]) for x in self.lines], dtype=np.float)
        avg = np.sum(descents_of_lines) / descents_of_lines.shape[0]

        arg = np.argsort(descents_of_lines - avg)
        avg = np.sum(descents_of_lines[arg[:5]])/5

        mat = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), atan(avg) * 180 / np.pi, 1)
        self.edges = cv2.warpAffine(self.edges, mat, (self.w * 2, self.h * 2))
        return self.edges

    def count(self):
        horizontal_sums = np.array([sum(x) for x in self.gray])
        self.descents = horizontal_sums[1:] - horizontal_sums[:-1]
        self.descents[np.where(self.descents < max(self.descents)/20)] = 0
        self.peaks = find_peaks(self.descents, distance=10)
        return self.peaks, self.descents, self.peaks[0].shape[0]


if __name__ == "__main__":
    img = cv2.imread('test.png')
    m = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), 90, 1)
    rot = cv2.warpAffine(img, m, (img.shape[0], img.shape[1]))
    counter = Counter(img)
    rotation = counter.rotation_img()
    cv2.imshow('1', counter.img)
    cv2.imshow('', rotation)
    peaks, descents, num = counter.count()
    peaks, _ = peaks
    print(peaks)
    #plt.subplot(2,2,1)
    plt.plot(peaks, descents[peaks], 'xr')
    plt.plot(descents)
    plt.title(num)

    plt.show()

