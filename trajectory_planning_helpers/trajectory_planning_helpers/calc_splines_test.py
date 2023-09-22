# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
import helper_funcs_glob
import os

# 散点通过等式条件计算各段之间的三次样条曲线系数，闭环非闭环均可
# required input: path, el_lengths=None, psi_s/e, use_dist_scaling=False

#Test Case 1 
# path = np.array([(0,-1,4,4), # -1 instead of 0 in case norvec cross
#                (2,2,4,4),
#                (4,4,4,4),
#                (6,6,4,4),
#                (10,8,4,4),
#                (15,8,4,4),
#                (20,8,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
# psi_s=math.pi/4
# psi_e=0
# el_lengths=None
# use_dist_scaling=False

#Test Case 2
#path = np.array([(0,0,4,4), 
#                (4,0,4,4),
#                (8,0,4,4),
#                (12,0,4,4),
#                (16,4,4,4),
#                (16,10,4,4),
#                (16,15,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
#psi_s=0
#psi_e=math.pi/2
#el_lengths=None
#use_dist_scaling=False

# #Test Case 3
# path = np.array([(0,-4,4,4), 
#                  (4,0,4,4),
#                  (9,0,4,4),
#                  (12,0,4,4)]) # the last one is the extension
# psi_s=math.pi/4
# psi_e=0
# el_lengths=None
# use_dist_scaling=False

#Test Case 1 
path = np.array([(0,-1,4,4), # -1 instead of 0 in case norvec cross
                 (6,4,4,4),
                 (10,8,4,4),
                 (20,8,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
psi_s= 0
psi_e= 0
el_lengths=None
use_dist_scaling=False

#Test Case 4
# imp_opts = {"flip_imp_track": False,                # flip imported track to reverse direction
#             "set_new_start": False,                 # set new starting point (changes order, not coordinates)
#             "new_start": np.array([0.0, -47.0]),    # [x_m, y_m], set new starting point
#             "min_track_width": None,                # [m] minimum enforced track width (set None to deactivate)
#             "num_laps": 1}   
# path = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
#                                                         file_path='../../inputs/tracks/shanghai.csv',
#                                                         width_veh=2.0)
# psi_s=math.pi/4
# psi_e=0
# el_lengths=None
# use_dist_scaling=False

# check if path is closed
if np.all(np.isclose(path[0], path[-1])):
    closed = True
else:
    closed = False

# check inputs
if not closed and (psi_s is None or psi_e is None):
    raise ValueError("Headings must be provided for unclosed spline calculation!")

if el_lengths is not None and path.shape[0] != el_lengths.size + 1:
    raise ValueError("el_lengths input must be one element smaller than path input!")

# if distances between path coordinates are not provided but required, calculate euclidean distances as el_lengths
# 计算欧氏距离
if use_dist_scaling and el_lengths is None:
    el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))
elif el_lengths is not None:
    el_lengths = np.copy(el_lengths)

# if closed and use_dist_scaling active append element length in order to obtain overlapping elements for proper
# scaling of the last element afterwards
if use_dist_scaling and closed:
    el_lengths = np.append(el_lengths, el_lengths[0])

# get number of splines
# 有多少条spline, 2个点一条spline，即比输入的点数少1
no_splines = path.shape[0] - 1

# calculate scaling factors between every pair of splines
if use_dist_scaling:
    scaling = el_lengths[:-1] / el_lengths[1:]
else:
    scaling = np.ones(no_splines - 1)

# ------------------------------------------------------------------------------------------------------------------
# DEFINE LINEAR EQUATION SYSTEM ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# M_{x,y} * a_{x,y} = b_{x,y}) with a_{x,y} being the desired spline param
# *4 because of 4 parameters in cubic spline 三次样条曲线有4个参数
M = np.zeros((no_splines * 4, no_splines * 4))
b_x = np.zeros((no_splines * 4, 1))
b_y = np.zeros((no_splines * 4, 1))

# create template for M array entries
# row 1: beginning of current spline should be placed on current point (t = 0)
# 第一行：等于当前t=0时的x(或y，取决于哪个参数方程)
# row 2: end of current spline should be placed on next point (t = 1)
# 第二行：等于当前t=1时的x(或y，取决于哪个参数方程)
# row 3: heading at end of current spline should be equal to heading at beginning of next spline (t = 1 and t = 0)
# 第三行：t=1时的导数应等于下一段t=0时的导数
# row 4: curvature at end of current spline should be equal to curvature at beginning of next spline (t = 1 and
#        t = 0)
# 第四行：t=1时的二阶导数应等于下一段t=0时的二阶导数
template_M = np.array(                          # current point               | next point              | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1_(i+1)             = 0
                [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2_(i+1)   = 0

for i in range(no_splines):
    j = i * 4

    if i < no_splines - 1:
        M[j: j + 4, j: j + 8] = template_M #如i=0时前4行，前1到4行1到8列，i=1时，5~8行5到12列

        M[j + 2, j + 5] *= scaling[i]
        M[j + 3, j + 6] *= math.pow(scaling[i], 2)

    else:
        # no curvature and heading bounds on last element (handled afterwards)
        M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],
                                [1,  1,  1,  1]]

    b_x[j: j + 2] = [[path[i,     0]], #x_i, 对应template_M的第一行
                        [path[i + 1, 0]]] #x_i+1， 对应template_M的第二行
    b_y[j: j + 2] = [[path[i,     1]], #y_i
                        [path[i + 1, 1]]] #y_i+1

# ------------------------------------------------------------------------------------------------------------------
# SET BOUNDARY CONDITIONS FOR LAST AND FIRST POINT -----------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

if not closed:
    # if the path is unclosed we want to fix heading at the start and end point of the path (curvature cannot be
    # determined in this case) -> set heading boundary conditions
    # 这种情况下，固定航向，不管曲率

    # heading start point
    M[-2, 1] = 1  # heading start point (evaluated at t = 0)

    if el_lengths is None:
        el_length_s = 1.0
    else:
        el_length_s = el_lengths[0]

    b_x[-2] = math.cos(psi_s ) * el_length_s #参数方程的斜率写法， 即y'/x' = sin(psi_e)/cos(psi_e)
    b_y[-2] = math.sin(psi_s ) * el_length_s

    # heading end point
    M[-1, -4:] = [0, 1, 2, 3]  # heading end point (evaluated at t = 1)

    if el_lengths is None:
        el_length_e = 1.0
    else:
        el_length_e = el_lengths[-1]

    b_x[-1] = math.cos(psi_e ) * el_length_e
    b_y[-1] = math.sin(psi_e ) * el_length_e

else:
    # heading boundary condition (for a closed spline)
    M[-2, 1] = -1*scaling[-1]
    M[-2, -3:] = [1, 2, 3]
    # b_x[-2] = 0
    # b_y[-2] = 0

    # curvature boundary condition (for a closed spline)
    M[-1, 2] = -2 * math.pow(scaling[-1], 2)
    M[-1, -2:] = [2, 6]
    # b_x[-1] = 0
    # b_y[-1] = 0

#print("====")
#print(b_x)

# ------------------------------------------------------------------------------------------------------------------
# SOLVE ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

x_les = np.squeeze(np.linalg.solve(M, b_x))  # squeeze removes single-dimensional entries
y_les = np.squeeze(np.linalg.solve(M, b_y))

# get coefficients of every piece into one row -> reshape
coeffs_x = np.reshape(x_les, (no_splines, 4))
coeffs_y = np.reshape(y_les, (no_splines, 4))

# get normal vector (behind used here instead of ahead for consistency with other functions) (second coefficient of
# cubic splines is relevant for the heading)
normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

# normalize normal vectors
norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
normvec_normalized = np.expand_dims(norm_factors, axis=1) * normvec

print("===coeffs_x===")
print(coeffs_x)
print("===coeffs_y===")
print(coeffs_y) 
print("===M===")
print(M)
print("===normvec===")
print(normvec_normalized)
#M即A矩阵

#------------result visulization------------#
row_n = len(coeffs_x[:,1])
loc_x_array = []
loc_y_array = []
for i in range(row_n): #某段spline
    t = np.array(range(0,11))*0.1 
    coeffs_x_c = coeffs_x[i,:]
    coeffs_y_c = coeffs_y[i,:]
    for k in t:
        loc_x = coeffs_x_c[0]+coeffs_x_c[1]*k+coeffs_x_c[2]*pow(k,2)+coeffs_x_c[3]*pow(k,3)
        loc_y = coeffs_y_c[0]+coeffs_y_c[1]*k+coeffs_y_c[2]*pow(k,2)+coeffs_y_c[3]*pow(k,3)
        loc_x_array.append(loc_x)
        loc_y_array.append(loc_y)

#-----结果Review------#
plt.figure()
plt.plot(loc_x_array, loc_y_array, 'k-',label = "Spline")
plt.plot(loc_x_array, loc_y_array, 'ko',label = "Spline Points")
plt.plot(path[:,0],path[:,1],'g-',label = "Org")
plt.plot(path[:,0],path[:,1],'go',label = "Org Points")
plt.xlabel("east in m")
plt.ylabel("north in m")
plt.legend()
plt.axis('equal')
plt.title('Spline Fitting Result')
plt.show()


