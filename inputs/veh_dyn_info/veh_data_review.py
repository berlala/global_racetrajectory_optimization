import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
import math 
import csv

# -------------------------------------------------------
# This script is used to review the vehicle configuration
# -------------------------------------------------------
# 对高性能轮胎和赛道，附着系数可以大于1。

# Generation friction cycle based on current velocity and max ax and ay
def get_current_v_ggv_limit(ax_limit, ay_limit, cur_v):
    current_ax_limit = np.array([])
    current_ay_limit = np.array([])
    ylist = []
    for i  in range(int(ax_limit+1)): # this will be processed later
        ylist.append(math.sqrt(1-(i*i)/(ax_limit*ax_limit))*ay_limit) 

    ax_1q = np.array(range(int(ax_limit+1)))
    ay_1q = np.array(ylist)
    current_ax_limit = np.append(current_ax_limit,ax_1q)
    current_ay_limit = np.append(current_ay_limit,ay_1q)
    ax_2q = np.flipud(ax_1q)
    ay_2q = np.flipud(-1.0*ay_1q)
    current_ax_limit = np.append(current_ax_limit,ax_2q[1:])
    current_ay_limit = np.append(current_ay_limit,ay_2q[1:])
    ax_3q = -1.0*ax_1q
    ay_3q = -1.0*ay_1q
    current_ax_limit = np.append(current_ax_limit,ax_3q[1:])
    current_ay_limit = np.append(current_ay_limit,ay_3q[1:])
    ax_4q = np.flipud(-1.0*ax_1q)
    ay_4q = np.flipud(ay_1q)
    current_ax_limit = np.append(current_ax_limit,ax_4q[1:])
    current_ay_limit = np.append(current_ay_limit,ay_4q[1:])

    current_v = np.ones(len(current_ay_limit))*cur_v

    return current_ax_limit, current_ay_limit, current_v


# GGV Data
csv_file_ggv = csv.reader(open('ggv_normal.csv'))
csv_file_motor = csv.reader(open('ax_max_machines.csv'))

v_ggv = [] # m/s
v_motor = []
ax_ggv_max = []
ay_ggv_max = []
ax_motor = []
index_1  = 0
index_2 = 0

for item in csv_file_ggv:
    if index_1 > 1:
        v_ggv.append(float(item[0]))
        ax_ggv_max.append(float(item[1]))
        ay_ggv_max.append(float(item[2]))
    index_1=index_1+1;

for item in csv_file_motor:
    if index_2 > 1:
        v_motor.append(float(item[0]))
        ax_motor.append(float(item[1]))
    index_2=index_2+1;

plt.plot(v_ggv, ax_ggv_max)
plt.plot(v_ggv, ay_ggv_max)
plt.plot(v_motor, ax_motor)
plt.legend(['Ax GGV Max','Ay GGV Max','Ax Powertrain'])
plt.xlabel('Spd[m/s]')
plt.ylabel('Acc[m/s^2]')
plt.show()

fig2 = plt.figure()
# data preparation
ax_limit = np.array([])
ay_limit = np.array([])
index_v = np.array([])

for i in range(len(v_ggv)):
    current_limit  = []
    [cur_ax,cur_ay, cur_v] = get_current_v_ggv_limit(ax_ggv_max[i], ay_ggv_max[i], v_ggv[i])
    index_v = np.append(index_v,cur_v)
    ax_limit = np.append(ax_limit,cur_ax)
    ay_limit = np.append(ay_limit,cur_ay)

ax = fig2.add_subplot(111, projection='3d')
ax.scatter(ax_limit, ay_limit, index_v)  # plot_surface requires a same size x,y
plt.title("GGV under Different Speed")
ax.set_xlabel('X-Acc[m/s]')
ax.set_ylabel('Y-Acc[m/s^2]')
ax.set_zlabel('Spd[m/s]')
# scatter  or plot_trisurf
plt.show()
