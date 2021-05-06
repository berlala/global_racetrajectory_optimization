# Compare Trajectory Result
# Bolin ZHAO

import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# 坐标旋转
def loc_rot(x_m,y_m,rot_angle):  # rot_angle 为旋转弧度
    x_r = []
    y_r = []
    for i in range(len(x_m)):
        x_r.append(x_m[i]*math.cos(rot_angle) - y_m[i]*math.sin(rot_angle))
        y_r.append(x_m[i]*math.sin(rot_angle) + y_m[i]*math.cos(rot_angle))
    return x_r,y_r

# 坐标平移
def loc_move(x_m,y_m,move_x,move_y):  # move_x,move_y 为平移距离
    x_mv = []
    y_mv = []
    for i in range(len(x_m)):
        x_mv.append(x_m[i] + move_x)
        y_mv.append(y_m[i] + move_y)
    return x_mv,y_mv

# 坐标缩放
def loc_zoom(x_m,y_m,zoom_x,zoom_y):  # zoom_x,zoom_y 为缩放倍数
    x_z = []
    y_z = []
    for i in range(len(x_m)):
        x_z.append(x_m[i]*zoom_x)
        y_z.append(y_m[i]*zoom_y)
    return x_z,y_z    

# Result 1: original 
# index: # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
csv_data = np.loadtxt('outputs/traj_race_cl.csv',comments='#',delimiter=';')
#csv_time = np.loadtxt('outputs/lap_time_matrix.csv',comments='#',delimiter=',') # 并不是圈速数据
csv_time = np.loadtxt('outputs/t_profile_cl.csv',comments='#',delimiter=',')
res_x = []
res_y = []
vel_org = []
s_org = []
t_org = []
for i in csv_data:
    res_x.append(i[1])
    res_y.append(i[2])
    vel_org.append(i[5])
    s_org.append(i[0])
for i in csv_time:
    t_org.append(i)

#print(t_org)


# Result 2: Bolin Curv
data = np.load('outputs/kanel_cl.npz')
kappa = data['kappa']
el_lengths = data['el_lengths']
race_x = data['raceline_x']
race_y = data['raceline_y']
vel_crv = data['vx_profile_opt']
t_crv = data['t_profile_cl']

# Result 2: Bolin Shorest
data = np.load('outputs/shorest_cl.npz')
race_sx = data['raceline_x']
race_sy = data['raceline_y']
vel_sht = data['vx_profile_opt']
t_sht = data['t_profile_cl']

# Track info (from Result 2)
bond_ux = data['bond_up_x']
bond_uy = data['bond_up_y']
bond_dx = data['bond_down_x']
bond_dy = data['bond_down_y']

# Result 3: Shanghai Velocity
data = np.load('outputs/shanghai_ctcc.npz')
t_ctcc = data['t']
spd_ctcc_kmh = data['spd_kmh']

# Result 4: Shanghai Lambo Data
data = np.load('outputs/shanghai_lambo.npz')
t_lamb = data['t']
spd_lamb_kmh = data['spd_kmh']
lamb_x = data['x_m']
lamb_y = data['y_m']
rot_angle = math.pi*1.315 # 越大越逆时针
lamb_xr,lamb_yr = loc_rot(lamb_x,lamb_y,rot_angle)
move_x = 715
move_y = 760
lamb_xmr, lamb_ymr = loc_move(lamb_xr,lamb_yr,move_x,move_y)
zoom_x = 0.8489
zoom_y = zoom_x
lamb_xzmr, lamb_yzmr = loc_zoom(lamb_xmr,lamb_ymr,zoom_x,zoom_y)

#Plot Result
plt.figure(1)
plt.plot(res_x,res_y)
plt.plot(race_x,race_y)
plt.plot(race_sx,race_sy)
plt.plot(lamb_xzmr,lamb_yzmr,'b')
plt.plot(bond_ux,bond_uy,'k')
plt.plot(bond_dx,bond_dy,'k')
plt.axis('equal')
plt.legend(['Origianl','wo smooth','Shortest','lamb'])

plt.figure(2)
plt.plot(t_org, np.multiply(vel_org,3.6))
plt.plot(t_crv[:-1],np.multiply(vel_crv,3.6))
plt.plot(t_sht[:-1],np.multiply(vel_sht,3.6))
plt.plot(t_ctcc[30:]-t_ctcc[30], spd_ctcc_kmh[30:])
plt.plot(t_lamb, spd_lamb_kmh)
plt.legend(['Origianl','wo smooth','Shortest','CTCC','Lamb'])
plt.xlabel('Time[s]')
plt.ylabel('Spd[km/h]')
plt.show()



