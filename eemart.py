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


class EEMART():
    def __init__(self , voc , idf , EM_weight , EEM_weight , event_list ,  gesture , image_path , event_threhold , predict_eventnum , episodic_num , loop_threhold , database ):
        self.voc = voc
        self.idf = idf
        self.EM_weight = EM_weight
        self.EEM_weight = EEM_weight
        self.event_list = event_list
        self.gesture = gesture
        self.image_path = image_path
        self.event_threhold = event_threhold
        self.predict_eventnum = predict_eventnum
        self.episodic_num = episodic_num
        self.loop_threhold = loop_threhold
        self.database = database
    def eemart(self):
        #参数预定义
        error = []  #坐标位置误差
        event_sim = [] #神经元相似度
        error_num = 0
        loop_num = 0
        last_em_node = 0
        em_map = nx.Graph()
        activate_episodic = np.zeros((1, 4))
        episodic_num_1 = 0
        loop = []
        event_num = []
        episodic_map = np.zeros((1, 4))
        image_path = self.image_path
        gesture = self.gesture
        event_threhold = self.event_threhold
        database = self.database
        voc = self.voc
        idf = self.idf
        EM_weight = self.EM_weight
        predict_eventnum = self.predict_eventnum
        episodic_num = self.episodic_num
        event_list = self.event_list
        loop_threhold = self.loop_threhold
        EEM_weight = self.EEM_weight
        numwords = 1000
        sift = cv2.xfeatures2d.SIFT_create()
        epi_num = []
        # 位姿信息预处理
        '''
        gesture[: , 0] = np.round(gesture[: , 0] , 2)
        gesture[: , 1] = np.round(gesture[: , 1] , 2)
        gesture[: , 2] = np.round(gesture[: , 2] , 1)
        gesture[: , 3] = np.round(gesture[: , 3] , 1)
        '''
        x = gesture[:, 0]
        y = gesture[:, 1]
        o_z = gesture[:, 4]
        o_w = gesture[:, 5]

        #开始训练
        for i in range(0 , len(x) , 3):
            event = 0
            # 输入数据预处理
            input_x = x[i]
            input_y = y[i]
            input_oz = o_z[i]
            input_ow = o_w[i]
            input_img = i
            input_img = str(input_img)
            img_name = input_img + ".jpg"
            img_path = image_path + img_name
            img = cv2.imread(img_path)
            #print(img_path)
            # 图像数据转化为词袋模型
            des_list = []
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            des_list.append((img_path, des))
            descriptors = des_list[0][1]

            test_features = np.zeros((1, numwords), "float32")
            words, distance = vq(descriptors, voc)
            for w in words:
                test_features[0][w] += 1
            test_features = test_features * idf
            test_features = preprocessing.normalize(test_features, norm='l2')
            # 输入向量归一化
            input_data = np.zeros(2008)

            input_data[0] = input_x
            input_data[1] = input_y
            input_data[2] = input_oz
            input_data[3] = input_ow
            input_data[4 : 1004] = -test_features
            input_data_1 = 1 - input_data[0 : 1004]

            input_data[1004: 2008] = input_data_1
            #print(input_data)
            # 事件节点ART判断

            node_num = np.zeros(EM_weight.shape[0])
            for j in range(EM_weight.shape[0]):
                min_and = 0
                for k in range(2008):
                    min_num = min(input_data[k], EM_weight[j][k])
                    min_and = min_and + min_num
                em_and = sum(EM_weight[j])
                node_num[j] = 0.5 * min_and / (0.1 + em_and)
            activate_node = np.argsort(-node_num)
            activate_node = activate_node[0]
            # signification
            min_and = 0
            for j in range(2008):
                min_num = min(input_data[j], EM_weight[activate_node][j])
                min_and = min_and + min_num
            print(min_and)
            em_and = sum(input_data)
            print(em_and)
            simlation = min_and / em_and
            print(simlation)
            event_sim.append(simlation)
            if (simlation < event_threhold):
                EM_weight = np.vstack((EM_weight, input_data))
                event = EM_weight.shape[0] - 1
                database = np.vstack((database, test_features))
                # database.append(test_features)
            else:
                EM_weight[activate_node, 0:3] = 0.05 * EM_weight[activate_node, 0:3] + 0.95 * input_data[0:3]
                event = activate_node


            #情景地图预定义
            em_node = np.zeros(predict_eventnum)
            em_node[event] = 1
            event_list_temp = event_list[1 : episodic_num , :]
            event_list[0 : episodic_num-1 , :] = event_list_temp
            event_list[episodic_num-1 , :] = em_node
            # 衰减计算
            '''
            event_list[0] = event_list[0] * 0.1
            event_list[1] = event_list[1] * 0.2
            event_list[2] = event_list[2] * 0.3
            event_list[3] = event_list[3] * 0.4
            event_list[4] = event_list[4] * 0.5
            event_list[5] = event_list[5] * 0.6
            event_list[6] = event_list[6] * 0.7
            event_list[7] = event_list[7] * 0.8
            event_list[8] = event_list[8] * 0.9
            event_list[9] = event_list[9] * 1.0
            '''

            input_list = event_list.reshape(episodic_num * predict_eventnum)
            # 计算情景地图
            node_num = np.zeros(EEM_weight.shape[0])
            em_activate_num = np.zeros(EEM_weight.shape[0])
            for j in range(EEM_weight.shape[0]):
                # 计算signification
                signification = 0
                min_and = 0
                sum_weight = 0
                for k in range(episodic_num * predict_eventnum):
                    em_min = min(input_list[k], EEM_weight[j][k])
                    min_and = min_and + em_min
                sum_weight = sum(EEM_weight[j])
                if (sum_weight == 0):
                    sum_weight = 1000
                signification = min_and / sum_weight

                # 计算similarity
                # 模值计算
                max_and = 0

                sim_up1 = np.dot(input_list, EEM_weight[j].T)
                sim_down1 = (np.linalg.norm(input_list) * np.linalg.norm(input_list)) * (
                        np.linalg.norm(EEM_weight[j]) * np.linalg.norm(EEM_weight[j]))
                if (sim_down1 == 0):
                    sim_down1 = 1000
                sim1 = sim_up1 / sim_down1
                # print("sim1" , sim1)

                for k in range(episodic_num * predict_eventnum):
                    em_max = max(input_list[k], EEM_weight[j][k])
                    max_and = max_and + em_max
                sim2 = min_and / max_and
                # print("sim2" , sim2)
                simliarity = (sim1 + sim2) / 2
                em_activate_num[j] = signification * (1 - (1 - simliarity))
                # print("em_activate_num" , em_activate_num)
            em_activate_node = np.argsort(-em_activate_num)[0]

            same_num = 0
            EEM_weight_temp = EEM_weight[em_activate_node].reshape((episodic_num , predict_eventnum))
            for i in range(episodic_num):
                if(EEM_weight_temp[i] == event_list[i]).all():
                    same_num += 1

            if(same_num > 5):

            #if (EEM_weight[em_activate_node] == input_list).all():
                # 激活先前情景，进行回环检测。
                memory = EEM_weight[em_activate_node].reshape(episodic_num , predict_eventnum)
                predict_event = 0
                activate_episodic[0, 0] = EM_weight[event][0]
                activate_episodic[0, 1] = EM_weight[event][1]
                predict_event = np.argsort(-memory[episodic_num-1])[0]
                # print(predict_event)
                # predict_event_1 = predict_event + 1
                scores = np.dot(test_features, database[predict_event].T)
                scores = np.max(scores)

                if (scores > loop_threhold):
                    loop_num += 1
                    error_num = error_num = math.sqrt(math.pow((EM_weight[predict_event , 0] - input_data[0]) , 2) + math.pow((EM_weight[predict_event , 1] - input_data[1]) , 2)) / 2
                    error.append(error_num)
                    EM_weight[predict_event , 0:3] =EM_weight[predict_event , 0:3]  + (input_data[0:3] - EM_weight[predict_event , 0:3]) / 2
                    #print("the line is", em_activate_node)
                    episodic_num_1 = EEM_weight.shape[0]
                    #print("loop closing", scores)
                    ux = abs(input_data[0] - EM_weight[predict_event][0])
                    uy = abs(input_data[1] - EM_weight[predict_event][1])
                    uroz = abs(input_data[2] - EM_weight[predict_event][2])
                    urow = abs(input_data[3] - EM_weight[predict_event][3])
                    urge = np.array([ux, uy, uroz, urow])
                    em_map.add_edge(last_em_node, em_activate_node)
                    last_em_node = em_activate_node
                else:
                    # 建立认知图，相关邻接列表，邻接矩阵
                    error_num = math.sqrt(math.pow((EM_weight[predict_event , 0] - input_data[0]) , 2) + math.pow((EM_weight[predict_event , 1] - input_data[1]) , 2))
                    error.append(error_num)
                    EEM_weight = np.vstack((EEM_weight, input_list))
                    #print("new eposidic", EEM_weight.shape[0])
                    #print("remember", i)
                    episodic_num_1 = EEM_weight.shape[0]
                    activate_episodic[0, 0] = EM_weight[event][0]
                    activate_episodic[0, 1] = EM_weight[event][1]
                    activate_episodic[0, 2] = EM_weight[event][2]
                    activate_episodic[0, 3] = EM_weight[event][4]
                    em_map.add_node(EEM_weight.shape[0])
                    em_map.add_edge(last_em_node, EEM_weight.shape[0])


            else:
                # 建立认知图，相关邻接列表，邻接矩阵
                error_num = math.sqrt(math.pow((EM_weight[event , 0] - input_data[0]) , 2) + math.pow((EM_weight[event , 1] - input_data[1]) , 2))
                error.append(error_num)
                # 新情景神经元编码完成，同时生成情景地图
                EEM_weight = np.vstack((EEM_weight, input_list))
                activate_episodic[0, 0] = EM_weight[event][0]
                activate_episodic[0, 1] = EM_weight[event][1]
                activate_episodic[0, 2] = EM_weight[event][2]
                activate_episodic[0, 3] = EM_weight[event][3]
                episodic_num_1 = EEM_weight.shape[0]
                em_map.add_node(EEM_weight.shape[0])
                em_map.add_edge(last_em_node, EEM_weight.shape[0])
                last_em_node = EEM_weight.shape[0]
                # 情景地图的生成

            # 情景地图神经元整合
            # C = np.array([input_list , input_x , input_y , input_oz , input_ow , input_data])
            # C1 = np.array([input_x , input_y , i])
            #print("remember", i)
            episodic_map = np.vstack((episodic_map, activate_episodic))
            np.save("/home/hzy/SSH/demo3/error" , error)
            epi_num.append(episodic_num_1)
            np.save("/home/hzy/SSH/demo3/episodic_map", episodic_map)
            np.save("/home/hzy/SSH/demo3/episodic", epi_num)
            loop.append(loop_num)
            np.save("/home/hzy/SSH/demo3/loop", loop)
            event_num.append(EM_weight.shape[0])
            np.save("/home/hzy/SSH/demo3/event", event_num)
            np.save("/home/hzy/SSH/demo3/event_sim" , event_sim)

        #nx.draw(em_map)
        #plt.savefig("D:/project/corridor_data/em_map.png")
        '''
        print(EEM_weight.shape)
        print(EM_weight.shape)
        print(error_num)
        print(loop_num)
        print(database.shape)
        '''
        return EM_weight , EEM_weight , database , error