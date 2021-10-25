import numpy as np
import pickle
from eemart import EEMART



#load version_data
voc = open(r"/home/hzy/SSH/demo3/voc64.pkl", "rb")
voc = pickle.load(voc)
idf = open(r"/home/hzy/SSH/demo3/idf64.pkl", "rb")
idf = pickle.load(idf)
print(idf)
#神经元预定义
predict_eventnum = 500
episodic_num = 7
EM_weight = np.zeros((1 , 2008))
EEM_weight = np.zeros((1 , 3500))
event_list = np.zeros((7 , 500))
loop_threhold = 0.95
database = np.ones((1, 1000))
event_threhold = 0.970

#训练阶段


train_gesture = np.loadtxt("/home/hzy/SSH/demo3//path")
#数据归一化
train_gesture[: , 1] = train_gesture[: , 1] + 4
train_gesture[: , 0] = train_gesture[: , 0] / 4
train_gesture[: , 1] = train_gesture[: , 1] / 4
train_gesture[: , 2] = 0
train_gesture[: , 3] = 0
train_gesture[: , 4] = (train_gesture[: , 4] + 1) / 2
train_gesture[: , 5] = (train_gesture[: , 5] + 1) / 2

#print(train_gesture)

train_image_path = "/home/hzy/SSH/demo3/image/"
EEMART0 = EEMART(voc , idf , EM_weight , EEM_weight , event_list ,  train_gesture , train_image_path , event_threhold , predict_eventnum , episodic_num , loop_threhold , database)
EM_weight , EEM_weight , database , error = EEMART0.eemart()
print(EM_weight.shape , EEM_weight.shape)

first_image_path = "/home/hzy/SSH/demo3/test_image/"
#first_gesture = np.loadtxt("D:/project/demo1/second_way/path")
first_gesture = np.loadtxt("/home/hzy/SSH/demo3/path")
EEMART0 = EEMART(voc , idf , EM_weight , EEM_weight , event_list ,  train_gesture , first_image_path , event_threhold , predict_eventnum , episodic_num , loop_threhold , database)
EM_weight , EEM_weight , database , error = EEMART0.eemart()
print(EM_weight.shape , EEM_weight.shape)
'''
train_image_path = "D:/project/demo2/image/"
train_gesture = np.loadtxt("D:/project/demo2/path")
EEMART0 = EEMART(voc , idf , EM_weight , EEM_weight , event_list ,  train_gesture , train_image_path , event_threhold , predict_eventnum , episodic_num , loop_threhold , database)
EM_weight , EEM_weight , database , error = EEMART0.eemart()
print(EM_weight.shape , EEM_weight.shape, database.shape )
'''




