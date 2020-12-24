import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from math import atan

img = cv2.imread('ukd2Y.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
qw = cv2.GaussianBlur(gray, (3, 3), 0, 0)
edges = cv2.Canny(gray, 80, 240, 3)

w, h = img.shape[0], img.shape[1]

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, 40, 15)


def cal_d(l):
    return (l[2] - l[0]) / (l[3] - l[1])


descents_of_lines = [cal_d(x[0]) for x in lines]

s = 0
for line in lines:
    line = line[0]
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    s += (y2 - y1) / (x2 - x1)

s /= len(lines)
M = cv2.getRotationMatrix2D((w / 2, h / 2), atan(s) * np.pi / 180, 1)
new = cv2.warpAffine(gray, M, (w * 2, h * 2))


def count(img):
    img = np.asarray(img)
    horizontal_sums = np.array([sum(x) for x in gray])
    descents = np.array([(horizontal_sums[x] - horizontal_sums[x - 1]) for x in range(1, len(horizontal_sums))])
    descents = horizontal_sums[1:] - horizontal_sums[0:-1]
    descents[np.where(descents < max(descents) / 10)] = 0
    return find_peaks(descents, distance=10), descents


p, descents = count(gray)
print(p[0].shape)
plt.plot(descents)
plt.show()
