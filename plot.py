import numpy as np
import matplotlib.pyplot as plt


#ERROR
error = np.load("/home/hzy/SSH/demo3/error.npy")
print(error.shape)
error = error[0:509] * 4
plt.plot(error)
plt.ylim([0 , 0.1])
plt.xlim([0 , 550])
plt.ylabel("error")
plt.xlabel("episodic")
plt.title("error")
plt.show()


#event_sim
event_sim = np.load("/home/hzy/SSH/demo3/event_sim.npy")
print(event_sim.shape)
plt.plot(event_sim)
plt.ylim([0.8 , 1])
plt.xlim([0 , 550])
plt.ylabel("event sim")
plt.xlabel("event")
plt.title("event sim")
plt.show()

#spisodic_map
episodic_map = np.load("/home/hzy/SSH/demo3/episodic_map.npy")
print(episodic_map.shape)
plt.plot(episodic_map)
#plt.ylim([0.8 , 1])
#plt.xlim([0 , 550])
#plt.ylabel("event sim")
#plt.xlabel("event")
plt.title("spisodic map")
plt.show()

#loop_num
loop = np.load("/home/hzy/SSH/demo3/loop.npy")
print(loop.shape)
a = loop.shape[0]
plt.plot(loop)
#plt.ylim([0.8 , 1])
#plt.xlim([0 , a])
plt.ylabel("loop num")
plt.xlabel("event")
plt.title("loop num")
plt.show()

#event num
event = np.load("/home/hzy/SSH/demo3/event.npy")
print(event.shape)
a = event.shape[0]
plt.plot(event)
#plt.ylim([0.8 , 1])
#plt.xlim([0 , a])
plt.ylabel("event num")
plt.xlabel("event")
plt.title("event num")
plt.show()


error_means = sum(error)/error.shape
event_sim_means = sum(event_sim) / event_sim.shape
print(error_means)
print(event_sim_means)

