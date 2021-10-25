#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from scipy.cluster.vq import *
from sklearn import preprocessing


numwords = 1000
sift = cv2.xfeatures2d.SIFT_create()


#voc_path = "D:/project/corridor_data/voc.pkl"
#idf_path = "D:/project/corridor_data/idf.pkl"
#database = "D:/project/corridor_data/database.pkl"
#img_lists = "D:/project/corridor_data/img_lists.pkl"

voc = open(r"/home/hzy/SSH/demo3/voc64.pkl" , "rb")
voc = pickle.load(voc)
idf = open(r"/home/hzy/SSH/demo3/idf64.pkl" , "rb")
idf = pickle.load(idf)
database = open(r"/home/hzy/SSH/database.pkl" , "rb")
database = pickle.load(database)
img_lists = open(r"/home/hzy/SSH/demo3/img_lists.pkl" , "rb")
img_lists = pickle.load(img_lists)
print(img_lists)
print(database.shape)


#检索图片数据预处理
img_path = "/home/hzy/SSH/demo3/image/82.jpg"
image = cv2.imread(img_path)
des_list = []
gray=cv2.cvtColor(image ,cv2.COLOR_RGB2GRAY)
kp, des = sift.detectAndCompute(gray, None)
des_list.append((img_path, des))
descriptors = des_list[0][1]



#print(descriptors)



#print(voc)



# 抽取图像Tf-Idf特征
test_features = np.zeros((1, numwords), "float32")
words, distance = vq(descriptors , voc)
for w in words:
    test_features[0][w] += 1
#print("test2" , test_features)
test_features = test_features * idf
#print("test_3" , test_features)
# 归一化
test_features = preprocessing.normalize(test_features, norm='l2')
#print(database.shape)
# 计算所有图像的相似度
scores = np.dot(test_features, database.T)

rank_ID = np.argsort(-scores)
print(np.sort(-scores))


scores_1 = np.zeros(1703)
print(test_features.shape)
c = sum(sum(test_features))
print(c)
for i in range(1703):
    a = 0
    b = 0
    for j in range(1000):
        #print(test_features.shape)
        #print(database[i][j])
        a = max(test_features[0 , j] , database[i , j])
        b = b + a
    scores_1[i] = b / c
print(np.sort(-scores_1))

scores_1 = np.zeros(1703)
print(test_features.shape)
print(database.shape)
test_features = -test_features
database = -database
test_features_1 = np.zeros((1 , 2000))
test_features_1[0 , 0:1000] = test_features[0 , :]
test_features_1[0 , 1000:2000] = 1 - test_features[0 , :]
c = sum(sum(test_features_1))
database_1 = np.zeros((1703 , 2000))
database_1[: , 0:1000] = database[: , :]
database_1[: , 1000:2000] = 1 - database[: , :]
for i in range(1703):
    a = 0
    b = 0
    for j in range(2000):
        #print(test_features.shape)
        #print(database[i][j])
        a = min(test_features_1[0 , j] , database_1[i , j])
        b = b + a
    scores_1[i] = b / c
print(np.sort(-scores_1))


'''
# 输出Top最相似的图像
for i, ID in enumerate(rank_ID[0][0:10]):
    #img = Image.open(img_lists[ID])
    print(ID)
    same_path = img_path + "/" + img_lists[ID]
    print(i)
    print(same_path)
'''




