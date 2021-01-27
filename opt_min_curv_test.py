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
# reftrack, normvectors, A, kappa_bound, w_veh

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
file_paths["track_name"] = "shanghai"    
file_paths["module"] = os.path.dirname(os.path.abspath(__file__))
file_paths["track_file"] = os.path.join(file_paths["module"], "inputs", "tracks", file_paths["track_name"] + ".csv")

reftrack_imp = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
                                                            file_path=file_paths["track_file"],
                                                            width_veh=2.0)

#print(reftrack)
#-----相当于pre_track------#
start_index = 60
end_index = 220
reftrack=reftrack_imp[start_index:end_index,:] #截取开环轨迹
#reftrack=reftrack_imp #原始闭环轨迹
#reftrack_interp = tph.spline_approximation.spline_approximation(track=reftrack_imp) #进行近似处理
reftrack_interp = reftrack  #不进行近似处理
refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_imp[end_index+1, 0:2]))  # 开环,补全最后一个
#refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0,0:2]))  #闭环，使收尾相等

##Show the original track 
#plt.plot(reftrack[:,0],reftrack[:,1])
#plt.axis('equal')
#plt.title('Original Track')
#plt.show()

# check if path is closed 判断是否闭环
if np.all(np.isclose(refpath_interp_cl[0], refpath_interp_cl[-1])):
    closed = True
    psi_s = None
    psi_e = None
    print("The track is closed.")
else:
    closed = False
    psi_s = 1/3.0*math.pi  # 开环必须提供
    psi_e = 0*math.pi
    print("The track is NOT closed.")

#print("The track is closed? " +str(closed))




#reftrack_interp, coeffs_y, A, coeffs_x, coeffs_y = helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=reftrack,
#                                                reg_smooth_opts=pars["reg_smooth_opts"],
#                                                stepsize_opts=pars["stepsize_opts"],
#                                                debug=debug,
#                                                min_width=imp_opts["min_track_width"])

[coeffs_x, coeffs_y, A, normvectors] = tph.calc_splines.calc_splines(path = refpath_interp_cl,el_lengths= None,
                                        psi_s= psi_s, psi_e= psi_e,use_dist_scaling=False)
#--------结束pretrack-------#

print_debug = False;
plot_debug = False;

kappa_bound = 0.2;
w_veh = 2.0;

# ------------------------------------------------------------------------------------------------------------------
# PREPARATIONS -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

reftrack = reftrack_interp #opt内部的reftrack, 和pretrack处的不一样

no_points = reftrack.shape[0]
print("===Matrix A===")
print(A)  #A矩阵是spline参数a,b,c,d的系数矩阵,由导数得到
print("===N Vectors===")
print(normvectors)
print("========")
print("num of points")
print(no_points)
print("norm vectors")
print(normvectors.shape[0]) #注意normvector的方向，可能会莫名反向
print("========")


# check inputs
if no_points != normvectors.shape[0]:
    raise ValueError("Array size of reftrack should be the same as normvectors!")

if no_points * 4 != A.shape[0] or A.shape[0] != A.shape[1]:
    raise ValueError("Spline equation system matrix A has wrong dimensions!")

# create extraction matrix -> only b_i coefficients of the solved linear equation system are needed for gradient
# information
A_ex_b = np.zeros((no_points, no_points * 4), dtype=np.int)

for i in range(no_points):
    A_ex_b[i, i * 4 + 1] = 1    # 1 * b_ix = E_x * x

#e.g., 当no_points = 3 时:
#A_ex_b = array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

# create extraction matrix -> only c_i coefficients of the solved linear equation system are needed for curvature
# information
A_ex_c = np.zeros((no_points, no_points * 4), dtype=np.int)

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
        M_x[j, i] = normvectors[i, 0]
        M_y[j, i] = normvectors[i, 1]
        M_x[j + 1, i + 1] = normvectors[i + 1, 0]
        M_y[j + 1, i + 1] = normvectors[i + 1, 1]
    else: #最后
        M_x[j, i] = normvectors[i, 0]
        M_y[j, i] = normvectors[i, 1]
        if closed is True:
            M_x[j + 1, 0] = normvectors[0, 0]  # close spline 
            M_y[j + 1, 0] = normvectors[0, 1] # close spline
        else:
            M_x[j + 1, 0] = 0  # open spline
            M_y[j + 1, 0] = 0 # open spline

print("===Martix Mx====")
print(M_x) # M_x是法向量矩阵，与待求解的alpha系数共同确定最优点在法向量方向上的位置

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

print("===b_x====")
print(q_x)

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

Q_x = np.matmul(curv_part, y_prime)
Q_y = np.matmul(curv_part, x_prime)

# this part is multiplied by alpha within the optimization (variable part)
E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)

# original curvature part (static part)
k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

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

if plot_debug:
    plt.plot(curv_orig_lin)
    plt.plot(curv_sol_lin)
    plt.legend(("original linearization", "solution based linearization"))
    plt.show()

# calculate maximum curvature error
curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))

# ------------------------------------------------------------------------------------------------------------------
# RESULT and PLOT --------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

print('==ratio (result)==')
print(alpha_mincurv)
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
for i in alpha_mincurv[int_index:]: #[:-1]和[-1]用法不一样，注意
    vec = normvectors[int_index]
    base = reftrack[int_index]
    track_x.append(base[0])
    track_y.append(base[1])

    bond_up_x.append(base[0] + vec[0]*-1*base[2])
    bond_up_y.append(base[1] + vec[1]*-1*base[2])
    bond_down_x.append(base[0] + vec[0]*+1*base[3])
    bond_down_y.append(base[1] + vec[1]*+1*base[3])

    result_x.append(vec[0]*i + base[0])
    result_y.append(vec[1]*i + base[1])

    int_index = int_index+1


plt.plot(track_x, track_y,'--',linewidth=0.6)
plt.plot(bond_up_x, bond_up_y)
plt.plot(bond_down_x, bond_down_y)
plt.plot(result_x,result_y,linewidth=1.5)
plt.legend(['Track Center','Up Bound','Down Bound','Opt Res'])
plt.xlabel("East[m]")
plt.ylabel("North[m]")
plt.title("Optimal Result")
#plt.plot(track_x, track_y,'o') # track center points
for i in range(len(track_x)):
    plt.plot([bond_down_x[i],track_x[i],bond_up_x[i]],[bond_down_y[i],track_y[i],bond_up_y[i]],'k--',linewidth=0.6)
plt.axis('equal')
plt.show()
