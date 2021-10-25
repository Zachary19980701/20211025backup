import numpy as np
import pickle
from eemart import EEMART



#load version_data
voc = open(r"D:/project/demo1/voc.pkl", "rb")
voc = pickle.load(voc)
idf = open(r"D:/project/demo1/idf.pkl", "rb")
idf = pickle.load(idf)
#神经元预定义
predict_eventnum = 200
episodic_num = 7
EM_weight = np.zeros((1, 72))
EEM_weight = np.zeros((1, 1400))
event_list = np.zeros((7, 200))
train_image_path = "D:/project/demo1/first_way/first_image/"
train_gesture = np.loadtxt("D:/project/demo1/first_way/path")
loop_threhold = 0.8
database = np.ones((1, 32))
event_threhold = 0.970

first_image_path = "D:/project/demo1/second_way/second_image/"
#first_gesture = np.loadtxt("D:/project/demo1/second_way/path")
first_gesture = np.loadtxt("D:/project/demo1/third_way/path")
EEMART0 = EEMART(voc , idf , EM_weight , EEM_weight , event_list ,  first_gesture , first_image_path , event_threhold , predict_eventnum , episodic_num , loop_threhold , database)
EM_weight , EEM_weight , database , error = EEMART0.eemart()
print(EM_weight.shape , EEM_weight.shape, database.shape )



#second_image_path = "D:/project/demo1/third_way/third_image/"
second_image_path = "D:/project/demo1/fifth_way/fifth_image/"
second_gesture = np.loadtxt("D:/project/demo1/third_way/path")
EEMART0 = EEMART(voc , idf , EM_weight , EEM_weight , event_list ,  second_gesture , second_image_path , event_threhold , predict_eventnum , episodic_num , loop_threhold , database )
EM_weight , EEM_weight , database , error = EEMART0.eemart()
print(EM_weight.shape , EEM_weight.shape, database.shape )


