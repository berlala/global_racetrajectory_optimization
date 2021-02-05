# Sequence Method   Ref DSCC2015 
import numpy as np
import math  
import matplotlib.pyplot as  plt


# Required Input: kappa, el_lengths
data = np.load('../../kanel_cl.npz') # 读取上赛道闭环轨迹结果
kappa = data['kappa']
el_lengths = data['el_lengths']
data_length = len(kappa)

mu = 1.0 # constant mu
grav = 9.8
v_max = 75 # [m/s]
mass = 1500

def _fcn_acc(v):
    m = 1500
    r = 0.25
    T_acc_engine = max(0,480-v*10)
    acc_max = T_acc_engine*13/(m*r)
    return acc_max

def _fcn_del(v):
    m = 1500
    g = 9.8
    mu = 1 - v*0.005
    F_brake = mu*m*g
    dec_max = F_brake/m
    return dec_max


vel_kappa = []
# first process: curvature limit
for i in kappa:
    vel_kappa.append(min(v_max, math.sqrt(mu*grav/abs(i))))

# temporary fix connection issue
if vel_kappa[5] == v_max and vel_kappa[-5]==v_max:
    vel_kappa[:5] = (v_max,)
    vel_kappa[-5:] = (v_max,)

# Double the track for avoid connection issue
vel_kappa_double = np.concatenate((vel_kappa,vel_kappa),axis = 0)
el_lengths_double = np.concatenate((el_lengths,el_lengths),axis = 0)

plt.plot(vel_kappa_double)

# second process: acceleration limit
vel_acc =[]
vel_acc.append(vel_kappa_double[0])

for i in range(len(vel_kappa_double)-1):
    vel_acc.append(min(vel_kappa_double[i+1], math.sqrt(pow(vel_acc[i],2) + 2*_fcn_acc(vel_acc[i])*el_lengths_double[i])))

plt.plot(vel_acc)

# third process: deceleration limit
vel_dec = vel_acc

for i in range(len(vel_acc)-1,1,-1):
    vel_dec[i-1] = min(vel_acc[i-1], math.sqrt(max(0.0, pow(vel_dec[i],2) + 2*_fcn_del(vel_dec[i])*el_lengths_double[i])) )

plt.plot(vel_dec)

# final process, rebuild
index_1 = int(data_length/2)
patch_1 = vel_dec[index_1:data_length]
patch_2 = vel_dec[data_length:data_length+index_1]
vel_rebuild = np.concatenate((patch_2,patch_1),axis=0)
if len(patch_1) + len(patch_2) != data_length:
    raise ValueError("length is not matched!")

plt.plot(vel_rebuild)
plt.legend(['Curv Cons','Acc cons','Dec cons','Final Res'])
plt.ylabel('Spd[m/s]')
plt.show()