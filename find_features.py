#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import *
from sklearn import preprocessing
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle

img_path = "/home/hzy/SSH/demo3/image/"
#读取图像文件名
img_lists = []
for filename in os.listdir(img_path):
    img_lists.append(filename)
print(img_lists)

numwords = 1000
sift = cv2.xfeatures2d.SIFT_create()
des_list = []
for i in range(0 , len(img_lists)):
    print(img_lists[i])
    img = cv2.imread(img_path + "/" + img_lists[i])

    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray,None)
    des_list.append((img_lists[i] , des))
print(des_list)
print(kp)
print(des)
descriptors = des_list[0][1]
print(des_list[0] , des_list[1])
print(descriptors)
for img_list , descriptor in des_list[1:]:
    print(img_list)
    print("descriptor" , descriptor.shape)
    descriptors = np.vstack((descriptors , descriptor))
print(descriptors.shape)
voc, variance = kmeans(descriptors, numwords, 1)
print(voc)

#图像视觉单词抽取
im_features = np.zeros((len(img_lists), numwords), "float32")
for i in range(0,len(img_lists)):
    #将图像的SIFT特征转换成视觉单词分布
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        #单张图像单词出现频次统计
        im_features[i][w] += 1

print(im_features)

#Tf-Idf计算
#统计idf
nbr_occurences = np.sum((im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(img_list)+1) / (1.0*nbr_occurences + 1)), 'float32')

print(nbr_occurences)

print(idf)

im_features = im_features * idf

print(im_features)
im_features = preprocessing.normalize(im_features, norm='l2')
print(im_features)

print(img_lists)

query = img_lists[356]
print(query)
path = img_path + "/" + query
print(path)
image = cv2.imread(path)
plt.imshow(image)
plt.show

im = cv2.imread(img_path + "/" + query)

des_list = []
gray=cv2.cvtColor(im ,cv2.COLOR_RGB2GRAY)
kp, des = sift.detectAndCompute(gray, None)
des_list.append((query, des))
descriptors = des_list[0][1]

print(des_list)
print(idf)

print(descriptors)

print(voc)

# 抽取图像Tf-Idf特征
test_features = np.zeros((1, numwords), "float32")
print("test1" , test_features)
words, distance = vq(descriptors , voc)
for w in words:
    test_features[0][w] += 1
print("test2" , test_features)
test_features = test_features * idf
print("test_3" , test_features)
# 归一化
test_features = preprocessing.normalize(test_features, norm='l2')

# 计算所有图像的相似度
scores = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-scores)

with open('/home/hzy/SSH/demo3/voc64.pkl', 'wb') as f:

    pickle.dump(voc, f)


with open('/home/hzy/SSH/demo3/idf64.pkl', 'wb') as f:

    pickle.dump(idf, f)




with open('/home/hzy/SSH/database.pkl', 'wb') as f:

    pickle.dump(im_features, f)




with open('/home/hzy/SSH/demo3/img_lists.pkl', 'wb') as f:

    pickle.dump(img_lists, f)

