import numpy as np
import trajectory_planning_helpers as tph
import sys
import matplotlib.pyplot as plt


min_width = None
reftrack_imp = np.array([(0,-1,4,4), # -1 instead of 0  in case norvec cross
                        (2,2,4,4),
                        (4,4,4,4),
                        (6,6,4,4),
                        (10,8,4,4),
                        (15,8,4,4),
                        (20,8,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
normvectors = np.array([(0.7071,-0.7071),
                        (0.7071,-0.7071),
                        (0.7071,-0.7071),
                        (0.7071,-0.7071),
                        (0,-1),
                        (0,-1),
                        (0,-1)])  # 如何避免弯心处的坐标交汇的问题？？？

# ------------------------------------------------------------------------------------------------------------------
# INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# smoothing and interpolating reference track
# 貌似在此处处理之后就自动闭环了，如果需要处理开环需要在此处进行处理
# spline_approximation只能用于闭环轨迹
#reftrack_interp = tph.spline_approximation.spline_approximation(track=reftrack_imp)
#因此取消spline处理
reftrack_interp = reftrack_imp

# calculate splines
#注意：轨迹长度不可变，必须为size+1(补1)
#refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2])) #首尾相连，闭环轨迹
refpath_interp_cl = np.vstack((reftrack_interp[:, :2],[21,8])) #开环轨迹,一定要补
print(reftrack_interp[:, :2],)
print('==')
print(refpath_interp_cl)

coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
calc_splines(path=refpath_interp_cl,el_lengths=None, psi_s=0, psi_e=0,
                    use_dist_scaling=False)
#coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
#    calc_splines(path=refpath_interp_cl,
#                    use_dist_scaling=False)
#输入散点，输出拟合轨迹的系列参数
#若输入的轨迹不是闭环轨迹，需要提供初始和结束点的psi,注意参数需要带上名称（如psi_s=0）
print('==')
print(normvec_normalized_interp)

# ------------------------------------------------------------------------------------------------------------------
# CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

print("length of normvec is " + str(len(normvec_normalized_interp)))
normals_crossing = tph.check_normals_crossing.check_normals_crossing(track=reftrack_interp,
                                                                        normvec_normalized=normvec_normalized_interp,
                                                                        horizon=3)

if normals_crossing:
    bound_1_tmp = reftrack_interp[:, :2] + normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 2], axis=1)
    bound_2_tmp = reftrack_interp[:, :2] - normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 3], axis=1)

    plt.figure()

    plt.plot(reftrack_interp[:, 0], reftrack_interp[:, 1], 'k-')
    for i in range(bound_1_tmp.shape[0]):
        temp = np.vstack((bound_1_tmp[i], bound_2_tmp[i]))
        plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

    plt.grid()
    ax = plt.gca()
    ax.set_aspect("equal", "datalim")
    plt.xlabel("east in m")
    plt.ylabel("north in m")
    plt.title("Error: at least one pair of normals is crossed!")

    plt.show()

    raise IOError("At least two spline normals are crossed, check input or increase smoothing factor!")

# ------------------------------------------------------------------------------------------------------------------
# ENFORCE MINIMUM TRACK WIDTH (INFLATE TIGHTER SECTIONS UNTIL REACHED) ---------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

manipulated_track_width = False

if min_width is not None:
    for i in range(reftrack_interp.shape[0]):
        cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]

        if cur_width < min_width:
            manipulated_track_width = True

            # inflate to both sides equally
            reftrack_interp[i, 2] += (min_width - cur_width) / 2
            reftrack_interp[i, 3] += (min_width - cur_width) / 2

if manipulated_track_width:
    print("WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation in"
            " order to match the requirements!", file=sys.stderr)

print("=track=")
print(reftrack_interp)
print("=normvec=")
print(normvec_normalized_interp) 
print("=a=")
print(a_interp) 
print("=coe_x=")
print(coeffs_x_interp) 
print("=coe_y=")
print(coeffs_y_interp)