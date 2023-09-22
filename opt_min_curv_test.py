import numpy as np
import math
import quadprog
#import cvxopt
import time
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
import helper_funcs_glob
import json
import os


# required input
# reftrack, normvectors_refline, A, kappa_bound, w_veh

#Test Case 1
#reftrack = np.array([(0,-1,4,4), # -1 instead of 0 in case norvec cross
#                    (2,2,4,4),
#                    (4,4,4,4),
#                    (6,6,4,4),
#                    (10,8,4,4),
#                    (15,8,4,4),
#                    (20,8,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
#refpath_interp_cl = np.vstack((reftrack[:, :2],[25,8])) 

#Test Case 2
#reftrack = np.array([(0,0,4,4), 
#                     (4,0,4,4),
#                     (6,0,4,4),
#                     (9,0,4,4),
#                     (14,2,4,4),
#                     (24,8,4,4),
#                     (30,13,4,4),
#                     (40,14,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
#refpath_interp_cl = np.vstack((reftrack[:, 0:2],[50,12])) 
# TODO:normverctor 与直接用calc_splines_test.py算出的有差异，差异来源是+math.pi/2导致的。
# 但是其对结果ratio不影响

imp_opts = {"flip_imp_track": False,                # flip imported track to reverse direction
            "set_new_start": False,                 # set new starting point (changes order, not coordinates)
            "new_start": np.array([0.0, -47.0]),    # [x_m, y_m], set new starting point
            "min_track_width": None,                # [m] minimum enforced track width (set None to deactivate)
            "num_laps": 1}   

file_paths = {"veh_params_file": "racecar.ini"}
file_paths["track_name"] = "shanghai"    # berlin_2018, shanghai,fridaytrack
file_paths["module"] = os.path.dirname(os.path.abspath(__file__))
file_paths["track_file"] = os.path.join(file_paths["module"], "inputs", "tracks", file_paths["track_name"] + ".csv")

reftrack_imp = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
                                                            file_path=file_paths["track_file"],
                                                            width_veh=2.0)

#print(reftrack)

#-----相当于pre_track-----------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
start_index = 100
end_index = 400
#reftrack=reftrack_imp[start_index:end_index,:] #[1]截取开环轨迹
reftrack=reftrack_imp #[1]原始闭环轨迹
reftrack_interp = tph.spline_approximation.spline_approximation(track=reftrack,
                                                                k_reg= 3,
                                                                s_reg= 10,
                                                                stepsize_prep= 5.0,
                                                                stepsize_reg= 7.0,
                                                                debug=False)    #[2]进行近似处理，注意spline_approximation会强行闭环
#reftrack_interp = reftrack  #[2]不进行近似处理
#refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_imp[end_index+1, 0:2]))  # [3]开环,补全最后一个
refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0,0:2]))  # [3]闭环，使收尾相等

#Show the original track 
plt.plot(reftrack[:,0],reftrack[:,1],'.')
plt.axis('equal')
plt.title('Original Track')
plt.show()

#-----end of pre_track-----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# check if path is closed 判断是否闭环
if np.all(np.isclose(refpath_interp_cl[0], refpath_interp_cl[-1])):
    closed = True
    psi_s = None
    psi_e = None
    print("The track "+ file_paths["track_name"] +" is closed.")
else:
    closed = False
    psi_s = math.atan((refpath_interp_cl[1,1] - refpath_interp_cl[0,1])/(refpath_interp_cl[1,0] - refpath_interp_cl[0,0])) # 开环必须提供首尾航向角
    psi_e = math.atan((refpath_interp_cl[-1,1] - refpath_interp_cl[-2,1])/(refpath_interp_cl[-1,0] - refpath_interp_cl[-2,0]))
    #print(psi_s/math.pi*180)
    #print(psi_e/math.pi*180)
    print("The track" + file_paths["track_name"] + " is NOT closed.")

#reftrack_interp, coeffs_y, A, coeffs_x, coeffs_y = helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=reftrack,
#                                                reg_smooth_opts=pars["reg_smooth_opts"],
#                                                stepsize_opts=pars["stepsize_opts"],
#                                                debug=debug,
#                                                min_width=imp_opts["min_track_width"])

[coeffs_x_refline, coeffs_y_refline, A, normvectors_refline] = tph.calc_splines.calc_splines(path = refpath_interp_cl,el_lengths= None,
                                        psi_s= psi_s, psi_e= psi_e,use_dist_scaling=False)
# Spline: x(t) = a + bt + ct^2 + dt^3
# 获得组成赛道的所有splines的x,y系数值(待定系数a,b,c,d)，spline固定系数矩阵A（待定系数前的，由导数关系得到的固定系数），法向量矩阵
#--------结束pretrack-------#

print_debug = False;
plot_debug = True; #显示曲率

kappa_bound = 0.2; #车辆转向曲率限制
w_veh = 2.0;

# ------------------------------------------------------------------------------------------------------------------
# PREPARATIONS -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

reftrack = reftrack_interp #opt内部的reftrack, 和pretrack处的不一样

no_points = reftrack.shape[0]
#print("===Matrix A===")
#print(A)  #A矩阵是spline参数a,b,c,d的系数矩阵,由导数得到
#print("===N Vectors===")
#print(normvectors_refline)
print("========")
print("num of input track points(reftrack)")
print(no_points)
print("num of cl points")
print(refpath_interp_cl.shape[0])
print("num of norm vectors")
print(normvectors_refline.shape[0]) #注意normvector的方向，可能会莫名反向
print("========")

# check inputs
if no_points != normvectors_refline.shape[0]:
    raise ValueError("Array size of reftrack should be the same as normvectors_refline!")

if no_points * 4 != A.shape[0] or A.shape[0] != A.shape[1]:
    raise ValueError("Spline equation system matrix A has wrong dimensions!")

# create extraction matrix -> only b_i coefficients of the solved linear equation system are needed for gradient
# information
A_ex_b = np.zeros((no_points, no_points * 4), dtype=int)

for i in range(no_points):
    A_ex_b[i, i * 4 + 1] = 1    # 1 * b_ix = E_x * x

#e.g., 当no_points = 3 时:
#A_ex_b = array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

# create extraction matrix -> only c_i coefficients of the solved linear equation system are needed for curvature
# information
A_ex_c = np.zeros((no_points, no_points * 4), dtype=int)

#print(np.size(A_ex_c,0)) 
#print(np.size(A_ex_c,1))

for i in range(no_points):
    A_ex_c[i, i * 4 + 2] = 2    # 2 * c_ix = D_x * x
#A_ex_c = array([[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]])

# invert matrix A resulting from the spline setup linear equation system and apply extraction matrix
A_inv = np.linalg.inv(A)   #参数A 矩阵从外部直接输入，这里只负责抽取
#print("==size fo A_inv===") 
#print(np.size(A_inv,0)) 
#print(np.size(A_inv,1))

T_c = np.matmul(A_ex_c, A_inv) # 矩阵乘法

# set up M_x and M_y matrices including the gradient information, i.e. bring normal vectors into matrix form
M_x = np.zeros((no_points * 4, no_points))
M_y = np.zeros((no_points * 4, no_points))

for i in range(no_points):
    j = i * 4

    if i < no_points - 1:
        M_x[j, i] = normvectors_refline[i, 0]
        M_y[j, i] = normvectors_refline[i, 1]
        M_x[j + 1, i + 1] = normvectors_refline[i + 1, 0]
        M_y[j + 1, i + 1] = normvectors_refline[i + 1, 1]
    else: #最后
        M_x[j, i] = normvectors_refline[i, 0]
        M_y[j, i] = normvectors_refline[i, 1]
        if closed is True:
            M_x[j + 1, 0] = normvectors_refline[0, 0]  # close spline 
            M_y[j + 1, 0] = normvectors_refline[0, 1] # close spline
        else:
            M_x[j + 1, 0] = 0  # open spline
            M_y[j + 1, 0] = 0 # open spline

#print("===Martix Mx====")
#print(M_x) # M_x是法向量矩阵，与待求解的alpha系数共同确定最优点在法向量方向上的位置

# set up q_x and q_y matrices including the point coordinate information
q_x = np.zeros((no_points * 4, 1))
q_y = np.zeros((no_points * 4, 1))

for i in range(no_points):
    j = i * 4

    if i < no_points - 1:
        q_x[j, 0] = reftrack[i, 0]
        q_x[j + 1, 0] = reftrack[i + 1, 0]

        q_y[j, 0] = reftrack[i, 1]
        q_y[j + 1, 0] = reftrack[i + 1, 1]
    else: #最后
        if closed is True:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[0, 0]
            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[0, 1]
        else: # 对开环数据的处理，注意此处需要最后延长点的信息和起点、终点heading信息
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = refpath_interp_cl[i+1, 0] #延长点的位置
            q_x[j + 2, 0] = math.cos(psi_s) # 起点heading
            q_x[j + 3, 0] = math.cos(psi_e) # 终点heading
            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = refpath_interp_cl[i+1, 1]
            q_y[j + 2, 0] = math.sin(psi_s)
            q_y[j + 3, 0] = math.sin(psi_e)

#print("===b_x====")
#print(q_x)

# set up P_xx, P_xy, P_yy matrices
# Hypothesis:基于段内heading（一阶导）变化不大的强假设
# 因此此处可直接用vec_x = A_ex*A^-1*vec_qx, 而不用类似(vec_qx+M_x*vec_alpha)引入待定系数alpha
x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x) # 乘单位阵对角化
#print("===x_dot====")
#print(x_prime)
#print("===x_dot from coeffs_x====") 
#print(np.eye(no_points, no_points)*coeffs_x[:,1])  # 即直接抽取z_x中bi,应与x_prime结果一致。注意z_x为[Num of Points,4]的矩阵
y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)

x_prime_sq = np.power(x_prime, 2)
y_prime_sq = np.power(y_prime, 2)
x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)

curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)    # calculate curvature denominator
curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                        where=curv_den != 0)                          # divide where not zero (diag elements)
curv_part_sq = np.power(curv_part, 2)

P_xx = np.matmul(curv_part_sq, y_prime_sq)
P_yy = np.matmul(curv_part_sq, x_prime_sq)
P_xy = np.matmul(curv_part_sq, x_prime_y_prime)

# ------------------------------------------------------------------------------------------------------------------
# SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

T_nx = np.matmul(T_c, M_x)
T_ny = np.matmul(T_c, M_y)

H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
H = H_x + H_xy + H_y
H = (H + H.T) / 2   # make H symmetric

f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
        + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
f = f_x + f_xy + f_y
f = np.squeeze(f)   # remove non-singleton dimensions

# ------------------------------------------------------------------------------------------------------------------
# KAPPA CONSTRAINTS ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# -k_bound <= k_ref + k_var <= k_bound 
# 其中k_bound 是车辆的最大转向等效曲率
# 可以理解为k_ref为车辆在通过路段处的默认转向角，k_var是法向量方向移动后的增量转向角
# 通过分解k_ref + k_var 

Q_x = np.matmul(curv_part, y_prime)  # y'/(x'^2+y'^2)^(3/2)
Q_y = np.matmul(curv_part, x_prime)  # x'/(x'^2+y'^2)^(3/2)

# this part is multiplied by alpha within the optimization (variable part)
E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)
#相当于x''和y''的含参部分的系数

# original curvature part (static part)
k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))
# from the curvature defination, Eq[9], 相当于x''和y''的常数部分

con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)  # multiplied by -1 as only LE conditions are poss.
con_stack = np.append(con_ge, con_le)

# ------------------------------------------------------------------------------------------------------------------
# CALL QUADRATIC PROGRAMMING ALGORITHM -----------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

"""
quadprog interface description taken from 
https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/quadprog_.py

Solve a Quadratic Program defined as:

    minimize
        (1/2) * alpha.T * H * alpha + f.T * alpha

    subject to
        G * alpha <= h
        A * alpha == b

using quadprog <https://pypi.python.org/pypi/quadprog/>.

Parameters
----------
H : numpy.array
    Symmetric quadratic-cost matrix.
f : numpy.array
    Quadratic-cost vector.
G : numpy.array
    Linear inequality constraint matrix.
h : numpy.array
    Linear inequality constraint vector.
A : numpy.array, optional
    Linear equality constraint matrix.
b : numpy.array, optional
    Linear equality constraint vector.
initvals : numpy.array, optional
    Warm-start guess vector (not used).

Returns
-------
alpha : numpy.array
        Solution to the QP, if found, otherwise ``None``.

Note
----
The quadprog solver only considers the lower entries of `H`, therefore it
will use a wrong cost function if a non-symmetric matrix is provided.
"""

# calculate allowed deviation from refline
dev_max_right = reftrack[:, 2] - w_veh / 2
dev_max_left = reftrack[:, 3] - w_veh / 2

# check that there is space remaining between left and right maximum deviation (both can be negative as well!)
if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
    raise ValueError("Problem not solvable, track might be too small to run with current safety distance!")

# consider value boundaries (-dev_max_left <= alpha <= dev_max_right)
G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
# np.eye()引入的是最大侧向位移约束
h = np.append(dev_max_right, dev_max_left)
h = np.append(h, con_stack)

# save start time
t_start = time.perf_counter()

# solve problem by CVXOPT -------------------------------------------------------------------------------------------
#args = [cvxopt.matrix(H), cvxopt.matrix(f), cvxopt.matrix(G), cvxopt.matrix(h)]
#sol = cvxopt.solvers.qp(*args)

#if 'optimal' not in sol['status']:
#    print("WARNING: Optimal solution not found!")
#alpha_mincurv = np.array(sol['x']).reshape((H.shape[1],))

# Or solve problem by quadprog ---------------------------------------------------------------------------------------
eigenvalue, featurevector = np.linalg.eig(H) #矩阵特征值，判断正定
#print("===Real Part of Eigen Value of A===")  # Eigen Debug
#print(eigenvalue.real)
#if eigenvalue.real.all() < 0:
#    print("A矩阵负定！")
alpha_mincurv = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

# print runtime into console window
if print_debug:
    print("Solver runtime opt_min_curv: " + "{:.3f}".format(time.perf_counter() - t_start) + "s")

# ------------------------------------------------------------------------------------------------------------------
# CALCULATE CURVATURE ERROR ----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# calculate curvature once based on original linearization and once based on a new linearization around the solution
q_x_tmp = q_x + np.matmul(M_x, np.expand_dims(alpha_mincurv, 1))
q_y_tmp = q_y + np.matmul(M_y, np.expand_dims(alpha_mincurv, 1))

x_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x_tmp)
y_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y_tmp)

x_prime_prime = np.squeeze(np.matmul(T_c, q_x) + np.matmul(T_nx, np.expand_dims(alpha_mincurv, 1)))
y_prime_prime = np.squeeze(np.matmul(T_c, q_y) + np.matmul(T_ny, np.expand_dims(alpha_mincurv, 1)))

curv_orig_lin = np.zeros(no_points)
curv_sol_lin = np.zeros(no_points)

for i in range(no_points):
    curv_orig_lin[i] = (x_prime[i, i] * y_prime_prime[i] - y_prime[i, i] * x_prime_prime[i]) \
                        / math.pow(math.pow(x_prime[i, i], 2) + math.pow(y_prime[i, i], 2), 1.5)
    curv_sol_lin[i] = (x_prime_tmp[i, i] * y_prime_prime[i] - y_prime_tmp[i, i] * x_prime_prime[i]) \
                        / math.pow(math.pow(x_prime_tmp[i, i], 2) + math.pow(y_prime_tmp[i, i], 2), 1.5)
#由参数方程曲率定义，Ref Paper[9]，获得连接处点的曲率

# calculate maximum curvature error
curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))

# ------------------------------------------------------------------------------------------------------------------
# RESULT and PLOT for Trajectory -----------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

#print('==ratio (result)==')
#print(alpha_mincurv)
#print(curv_error_max)

# Plot the final result on the global frame
result_pos = []
result_x = []
result_y = []
track_x = []
track_y = []
bond_up_x = []
bond_up_y = []
bond_down_x = []
bond_down_y = []
int_index = 0  #
raceline = []
for i in alpha_mincurv[int_index:]: #[:-1]和[-1]用法不一样，注意
    vec = normvectors_refline[int_index]
    base = reftrack[int_index]
    track_x.append(base[0]) # center x 
    track_y.append(base[1]) # center y 

    bond_up_x.append(base[0] + vec[0]*-1*base[2]) # center_x+ (-1)*normal_x*w_l
    bond_up_y.append(base[1] + vec[1]*-1*base[2])
    bond_down_x.append(base[0] + vec[0]*+1*base[3])
    bond_down_y.append(base[1] + vec[1]*+1*base[3])

    result_x.append(vec[0]*i + base[0])
    result_y.append(vec[1]*i + base[1])
    raceline.append([result_x[-1],result_y[-1]])

    int_index = int_index+1

raceline = np.asarray(raceline) # List转np.rray
result_x = np.asarray(result_x) # 转array为散点彩图
result_y = np.asarray(result_y)

plt.figure(1)
plt.plot(track_x, track_y,'--',linewidth=0.6)
plt.plot(bond_up_x, bond_up_y)
plt.plot(bond_down_x, bond_down_y)
plt.plot(result_x,result_y,linewidth=1.5)
plt.legend(['Track Center','Up Bound','Low Bound','Opt Res'])
plt.xlabel("East[m]")
plt.ylabel("North[m]")
plt.title("Optimal Path Result")
#plt.plot(track_x, track_y,'o') # track center points
for i in range(len(track_x)):
    plt.plot([bond_down_x[i],track_x[i],bond_up_x[i]],[bond_down_y[i],track_y[i],bond_up_y[i]],'k--',linewidth=0.6)
plt.axis('equal')
#Add Colored Scatter Points For Position Reference, 增加定位点用于Debug
color_sample_size = int(20) 
index_point_color  = np.multiply(np.array(range(math.floor(len(result_y)/color_sample_size))),color_sample_size)
cValue = np.random.rand(len(index_point_color),3)
plt.scatter(result_x[index_point_color], result_y[index_point_color], c=cValue,marker='s')
#plt.show()

if plot_debug:
    plt.figure(9)
    plt.plot(curv_orig_lin)
    plt.plot(curv_sol_lin)
    plt.scatter(index_point_color,curv_orig_lin[index_point_color], c=cValue,marker='s') # marker = squrare/x
    plt.title('Curvature Review')
    plt.legend(("original linearization", "solution based linearization"))

print('== == result waypoint num == ==')
print(np.size(result_x))

# ------------------------------------------------------------------------------------------------------------------
# CALCULATE SPEED --------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# 基于优化结果alpha构建赛车线，将其处理成与refline一致的格式
#psi_vel_opt, kappa_opt = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=coeffs_x_opt,
#                      coeffs_y=coeffs_y_opt,
#                      ind_spls=spline_inds_opt_interp,
#                      t_spls=t_vals_opt_interp)

#-----相当于pre_track-----------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
raceline_interp = raceline  #不进行近似处理
if closed is True:
    raceline_interp_interp_cl = np.vstack((raceline_interp[:, :2], raceline_interp[0,0:2]))  #闭环，使收尾相等
else:
    raceline_interp_interp_cl = np.vstack((raceline_interp[:, :2], reftrack_imp[end_index+1, 0:2]))  # 开环,补全最后一个

#-----end of pre_track-----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#对raceline,重新进行样条曲线拟合处理，获取参数
[coeffs_x_raceline, coeffs_y_raceline, A, normvectors_raceline] = tph.calc_splines.calc_splines(path = raceline_interp_interp_cl,el_lengths= None,
                                        psi_s= psi_s, psi_e= psi_e,use_dist_scaling=False)

# calculate new spline lengths
spline_lengths_raceline = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x_raceline,
                        coeffs_y=coeffs_y_raceline)
print('==size of origin spline length ==')
print(np.size(spline_lengths_raceline)) # size check ok


# interpolate splines for evenly spaced raceline points, 通过系数还原轨迹，此处操作会使index发生变化,减小stepsize_approx会使输出size变大
raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.\
    interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                    coeffs_x=coeffs_x_raceline,
                                    coeffs_y=coeffs_y_raceline,
                                    incl_last_point=False,
                                    stepsize_approx=1.0)

print('==size of interp raceline length ==')
print(np.size(t_values_raceline_interp)) # size check ok

# calculate element lengths
s_tot_raceline = float(np.sum(spline_lengths_raceline))
el_lengths_raceline_interp = np.diff(s_raceline_interp)
if closed is True:
    el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])
else:
    el_lengths_raceline_interp_cl = el_lengths_raceline_interp

print("====t_values_raceline_interp====")
print(np.size(t_values_raceline_interp)) #t的总长度，t为1维,0<t<1
#print(t_values_raceline_interp)

# calculate heading and curvature, 对生成的raceline,计算各点的航向和曲率
psi_vel_opt, kappa_opt = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=coeffs_x_raceline,
                      coeffs_y=coeffs_y_raceline,ind_spls=spline_inds_raceline_interp,
                      t_spls=t_values_raceline_interp)


#print('== == kappa size == ==')
#print(np.size(kappa_opt))

ggv, ax_max_machines = tph.import_veh_dyn_info.\
    import_veh_dyn_info(ggv_import_path='./inputs/veh_dyn_info/ggv.csv',
                        ax_max_machines_import_path='./inputs/veh_dyn_info/ax_max_machines.csv')

vx_profile_opt = tph.calc_vel_profile.calc_vel_profile(ggv=ggv,
                        ax_max_machines=ax_max_machines,
                        v_max=70,
                        kappa=kappa_opt,
                        el_lengths=el_lengths_raceline_interp_cl,
                        closed=closed,
                        dyn_model_exp=1.0,
                        drag_coeff=0.75,
                        v_start = 20,
                        m_veh=1200)

# calculate longitudinal acceleration profile
if closed is True:  
    vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
else:
    vx_profile_opt_cl = vx_profile_opt

ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                    el_lengths=el_lengths_raceline_interp_cl,
                                                    eq_length_output=False)

# calculate laptime
t_profile_cl = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_opt,
                                                ax_profile=ax_profile_opt,
                                                el_lengths=el_lengths_raceline_interp_cl)

# # Additional: 2015DSCC method
vx_profile_dscc = tph.seq_vel_profile.seq_vel_profile(kappa  = kappa_opt, 
                                                     el_lengths = el_lengths_raceline_interp_cl) #TODO:修复DSCC方法在开环中不能计算的问题
if closed is True:  
    vx_profile_dscc_cl = np.append(vx_profile_dscc, vx_profile_dscc[0])
else:
    vx_profile_dscc_cl = vx_profile_dscc

ax_profile_dscc = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_dscc_cl,
                                                    el_lengths=el_lengths_raceline_interp_cl,
                                                    eq_length_output=False)
t_profile_dscc = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_dscc_cl,
                                                ax_profile=ax_profile_dscc,
                                                el_lengths=el_lengths_raceline_interp_cl) 
# -----End of DSCC --------

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(s_raceline_interp, vx_profile_dscc)
plt.plot(s_raceline_interp, vx_profile_opt)
plt.xlabel('Distance[m]')
plt.ylabel('Spd[m/s]')
plt.legend(['DSCC','Origin'])
plt.title('Velocity Result Compare')
plt.subplot(2,1,2)
plt.plot(t_profile_dscc, vx_profile_dscc_cl)
plt.plot(t_profile_cl, vx_profile_opt_cl)
plt.xlabel('time[s]')
plt.ylabel('Spd[m/s]')
#plt.show()

# == Lap Time == 
laptime_origin_part = math.modf(max(t_profile_cl))
laptime_dscc_part = math.modf(max(t_profile_dscc))

print('== == == Lap Time == == ==')
print('Lap Time: '+ str(int(laptime_origin_part[1]/60)) + ' min ' + 
        str(int(laptime_origin_part[1]%60))+' sec '+str(int(1000*round(laptime_origin_part[0],3))))
print('DSCC Method Lap Time: '+ str(int(laptime_dscc_part[1]/60)) + ' min ' + 
         str(int(laptime_dscc_part[1]%60))+' sec '+str(int(1000*round(laptime_dscc_part[0],3))))

# ------------------------------------------------------------------------------------------------------------------
# RESULT and PLOT for Velocity -------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#print(np.size(t_profile_cl)) # debug for the size match for plot
#print(np.size(vx_profile_opt_cl))
#print(np.size(ax_profile_opt))
#print(index_point_color)

# velcity result compare
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t_profile_cl,vx_profile_opt_cl*3.6)
plt.title('Velocity Detial')
extend_size = len(t_profile_cl)/len(result_x) # 不能再此处转成整数，会造成较大错位
#print(extend_size)
ax1.scatter(t_profile_cl[(np.multiply(index_point_color,extend_size)).astype(int)], vx_profile_opt_cl[(np.multiply(index_point_color,extend_size)).astype(int)]*3.6, c=cValue,marker='s') # marker = squrare/x
ax2.plot(t_profile_cl[:-1],ax_profile_opt,'k',linewidth=0.5)
ax1.set_ylabel('Velocity[km/h]')
ax2.set_ylabel('Acc[m/s^2]')
ax1.set_xlabel('time[s]')
plt.legend(['Acc'])
#plt.show()

## 注意上图中不能用int(), math.floor()之类，而需要用.astype(np.int)进行类型转化。

# --------------------------------------------
# Plot The Race Trajectory with Velocity Info
# --------------------------------------------
#print('==check interp size == ')
#print(np.size(raceline_interp[:,0]))
#print(np.size(vx_profile_opt_cl))
plt.figure(4)  # 轨迹线带速度信息
cm = plt.cm.get_cmap('cool')
plt.plot(track_x, track_y,'--',linewidth=0.6)
plt.plot(bond_up_x, bond_up_y)
plt.plot(bond_down_x, bond_down_y)
sc = plt.scatter(raceline_interp[:,0],raceline_interp[:,1], s= 6, c = vx_profile_opt*3.6, cmap = cm)
cbar = plt.colorbar(sc) #添加速度颜色条
cbar.set_label('Spd[km/h]') # 颜色条的单位
plt.legend(['Track Center','Up Bound','Low Bound','Opt Res'])
plt.xlabel("East[m]")
plt.ylabel("North[m]")
plt.title("Optimal Result with Speed for Min Curv")
plt.axis('equal')
plt.show()

# 保存最小曲率法结果 
np.savez('outputs/kanel_cl.npz',kappa=kappa_opt,el_lengths = el_lengths_raceline_interp_cl,
                       raceline_x = result_x, raceline_y = result_y,
                       bond_up_x = bond_up_x, bond_up_y = bond_up_y,
                       bond_down_x =bond_down_x, bond_down_y = bond_down_y,
                       vx_profile_opt = vx_profile_opt, t_profile_cl = t_profile_cl)