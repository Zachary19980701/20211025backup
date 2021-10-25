import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import *
from sklearn import preprocessing
from PIL import Image
import os
import matplotlib
import math
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
import networkx as nx
from networkx.algorithms import approximation as approx
sift = cv2.xfeatures2d.SIFT_create()
imgae1 = cv2.imread('/home/hzy/SSH/demo2/image/554.jpg')
image2 = cv2.imread('/home/hzy/SSH/demo2/image/1424.jpg')
img1_path = '/home/hzy/SSH/demo2/image/554.jpg'
img2_path = '/home/hzy/SSH/demo2/image/1424.jpg'
numwords = 64
voc = open(r"/home/hzy/SSH/demo2/voc64.pkl", "rb")
voc = pickle.load(voc)
idf = open(r"/home/hzy/SSH/demo2/idf64.pkl", "rb")
idf = pickle.load(idf)

des_list = []
gray = cv2.cvtColor(imgae1, cv2.COLOR_RGB2GRAY)
kp, des = sift.detectAndCompute(gray, None)
des_list.append((img1_path, des))
descriptors = des_list[0][1]

test_features = np.zeros((1, numwords), "float32")
words, distance = vq(descriptors, voc)
for w in words:
    test_features[0][w] += 1
test_features1 = test_features * idf
test_features1 = preprocessing.normalize(test_features, norm='l2')

des_list = []
gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
kp, des = sift.detectAndCompute(gray, None)
des_list.append((img2_path, des))
descriptors = des_list[0][1]

test_features = np.zeros((1, numwords), "float32")
words, distance = vq(descriptors, voc)
for w in words:
    test_features[0][w] += 1
test_features2 = test_features * idf
test_features2 = preprocessing.normalize(test_features, norm='l2')

score = np.dot(test_features1 , test_features2.T)
final = np.zeros(64)
print(score)
print(test_features1)
for i in range(64):
    final[i] = min(test_features1[0 , i] , test_features2[0 , i])
score2 = sum(final) / sum(sum(test_features1))

print(score , score2)