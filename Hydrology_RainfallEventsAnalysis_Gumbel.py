import sys
sys.path.append("../0_PythonPackage_JH")
import numpy as np
import pandas as pd
from numpy.random import gumbel
import matplotlib.pyplot as plt
import sympy
# S.EulerGamma.n(10)
import PlotControl as plot
# curve fitting
from scipy.optimize import curve_fit, root, fsolve
import ExcelControl as excelCtl
from glob import glob
import DataControlTIFF as DCTif
# The cumulative distribution form of Gumbel model (Grelle et al.,
# 2013; Lee et al., 2020) (formual 2.)
def model_gumbel_cum_dis(para_x,para_lambda,para_eta):
    temone= -1*(para_x-para_lambda)/para_eta
    temone = np.array(temone).astype(float)
    temtwo = -1.0*np.exp(temone)
    F_sub_GUM = np.exp(temtwo)
    return F_sub_GUM
# the return period for a specified rainfall event (formual 3a.)
def calculate_T_return_period(F_sub_GUM):
    T = 1/(1-F_sub_GUM)
    return T

# according to the list of maximum rainfall with sort of ascending ,return br_sup_mao (formual 5.)
def calculate_br_sup_mao(list_max_rainfall_asc, para_r):
    list_items_onestep = np.linspace(para_r+1, len(list_max_rainfall_asc), len(list_max_rainfall_asc)-para_r)
    tem_one = 0.0
    for i in range(para_r):
        if i ==0:
            tem_one = (list_items_onestep-(i+1))/(len(list_max_rainfall_asc)-(i+1))
        else:
            tem_one = tem_one*(list_items_onestep-(i+1))/(len(list_max_rainfall_asc)-(i+1))
    # 乘以最大降雨数列
    tem_two = list_max_rainfall_asc[para_r:]
    list_tem_three = tem_one*tem_two
    br_sup_mao = np.sum(list_tem_three)/len(list_max_rainfall_asc)
    return br_sup_mao

# Return the arithmetic mean of an ascending list of maximum rainfall sequence. (formual 6.)
def calculate_b0_sup_mao(list_max_rainfall_asc):
    n_lenth = len(list_max_rainfall_asc)
    b0_sup_mao = np.sum(list_max_rainfall_asc)/n_lenth
    return b0_sup_mao

# according to the list of maximum rainfall with sort of ascending ,return b1_sup_mao (formual 7.)
def calculate_b1_sup_mao(list_max_rainfall_asc):
    list_items_onestep = np.linspace(1, len(list_max_rainfall_asc),len(list_max_rainfall_asc))
    if len(list_max_rainfall_asc)<=1:
        print("降序序列数据组长度有问题！")
        return
    list_items_twostep = (list_items_onestep - 1)/(len(list_max_rainfall_asc)-1)
    list_items_threestep = list_items_twostep * list_max_rainfall_asc
    n_lenth = len(list_max_rainfall_asc)
    b1_sup_mao = np.sum(list_items_threestep)/n_lenth
    return b1_sup_mao

# return eta_sup_mao (formual 8.)
def calculate_eta_sup_mao(b0_sup_mao,b1_sup_mao):
    eta_sup_mao = (2*b1_sup_mao-b0_sup_mao)/np.log(2.0)
    return eta_sup_mao

# return lambda_sup_mao (formual 9.)
def calculate_lambda_sup_mao(eta_sup_mao,b0_sup_mao,gamma):
    lambda_sup_mao = b0_sup_mao-gamma*eta_sup_mao
    return lambda_sup_mao

# Calculating the decomposed rainfall sequence. (formual 10.)
def generate_list_rainfall_downscaled(para_Q_ini,para_a0, para_H, Dr, para_int_k):
    """
    :param para_Q_ini:累积降雨量
    :param para_a0:由降雨数据率定的参数a0
    :param para_H:由降雨数据率定的参数H
    :param Dr:累积雨量的降雨历时
    :param para_int_k:降雨量拆分的次数
    :return:拆分后的降雨量序列
    """
    list_W = []
    # Obtaining the weight values W for different time scales.
    for index in range(para_int_k):
        para_t_timescale = Dr / (2 ** (para_int_k - index))  # 时间分辨率
        tem_a = calculate_a_t(para_a0, para_H, para_t_timescale)  # 不同时间分辨率下的参数a
        if index ==0:
            list_W = generate_W_list_sup_ij(tem_a, para_int_k-index)
        else:
            list_W_sub_one = generate_W_list_sup_ij(tem_a, para_int_k - index)
            list_W_sub_one_repeat = np.repeat(list_W_sub_one,2**(index))
            list_W = list_W*list_W_sub_one_repeat
    # Generating the decomposed rainfall sequence.
    list_rainfall_downscaled = para_Q_ini*list_W
    return list_rainfall_downscaled

# Generating a random sequence of weight values W for the corresponding temporal resolution
# based on a probability distribution.(formual 11.)
def generate_W_list_sup_ij(para_a_t, para_int_k):
    """
    :param para_a_t:时间分辨率为Dr/2^k，对应的参数a
    :param para_int_k:差分次数
    :return: 返回时间分辨率为Dr/2^k，长度为2^k的W权重序列
    """
    W_list_sup_ij = np.random.beta(para_a_t, para_a_t, 2**para_int_k)
    return W_list_sup_ij

# Nonlinear fitting, formula to be fitted.
def func(t, a0, H):
    return a0 * (t ** (-1*H))


# Calculating the parameter "a" for different temporal resolutions. (formual 12.)
def calculate_a_t(para_a0, para_H, para_t_timescale):
    """
    :param para_a0:由降雨数据率定的参数a0
    :param para_H:由降雨数据率定的参数H
    :param para_t_timescale:时间分辨率，即：Dr/2^k
    :return:当前时间分辨率下的参数a
    """
    a_t = para_a0*para_t_timescale**(-1*para_H)
    return a_t


# Estimating the value of a based on the adjacent weight matrix of the rainfall sequence. (formual 13.)
def estimate_a_with_W(W_list):
    """
    :param W_list:根据降雨序列的相邻权重矩阵
    :return:a的估计值
    """
    a_mao = 1/(8*np.var(W_list))-0.5
    return a_mao

# By using rainfall sequences of different time scales, solving for the weight sequence W lays the foundation for
# the subsequent determination of parameter a. (formual 14.)
def generate_W_list(rainfall_list):
    """
    :param rainfall_list: 不同时间分辨率的降雨序列
    :return: 权重序列，用于率定a0和H参数
    """
    list_list_jt = rainfall_list[0:len(rainfall_list):2]
    list_list_jt_add1 = rainfall_list[1:len(rainfall_list):2]
    W_list = list_list_jt/(list_list_jt+list_list_jt_add1)
    # Excluding invalid and zero values, which means retaining weight values greater than 0.
    list_index = np.where(W_list>0)
    W_list_fianal = W_list[list_index]
    return W_list_fianal

# obtain the file path of the tif data
def get_precFileName(tem_preStr, dirPath, input_ID_Str):
    # Z_SURF_C_BABJ_20210601002546_P_CMPA_RT_BCCD_0P01_HOR-PRE-2021060100.tif
    dirPath = dirPath # E:\JH\4 【论文写作】\【SCI】\【2 不同来源的降雨数据对泥石流危险性预测精度的影响_凉山地区】\0_Data\3_Prec\Pre_DPS_2021\pre_box #
    ID_Str = input_ID_Str # "2021060100.tif"
    tem_preStr = tem_preStr # "Z_SURF_C_BABJ_"
    # path = r'E:\dose_data\train\20181208'
    pathrtss1 = glob(f'{dirPath}/'+tem_preStr+'*'+ID_Str+'*.tif', recursive=True)
    if len(pathrtss1)==0:
        pathrtss ="nosuchfile"
    else:
        pathrtss = pathrtss1[0]
    return pathrtss



# Using the bisection method to solve for the cumulative extreme rainfall values corresponding to the desired return periods.
def binary_search(f, a, b, tol,lambda_sup_mao, eta_sup_mao,ideal_T):
    """
    Implements the binary search algorithm to find a root of the function f.
    Parameters:
    f (function): the function to find the root of
    a (float): the left endpoint of the search interval
    b (float): the right endpoint of the search interval
    tol (float): the tolerance for the root

    Returns:
    float: the root of the function
    """
    fa, fb = f(a,lambda_sup_mao, eta_sup_mao,ideal_T), f(b,lambda_sup_mao, eta_sup_mao,ideal_T)
    assert fa * fb < 0, "The function must change sign over the interval [a, b]."

    while b - a > tol:
        c = (a + b) / 2
        fc = f(c,lambda_sup_mao, eta_sup_mao,ideal_T)
        if fc == 0:
            return c
        elif fc * fa < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2

# Generating a function of T using the bisection method.
def function_gumbel_line(list_x, lambda_sup_mao, eta_sup_mao,ideal_T):
    F_sub_GUM = model_gumbel_cum_dis(list_x, lambda_sup_mao, eta_sup_mao)
    list_T = calculate_T_return_period(F_sub_GUM)-ideal_T
    return list_T

if __name__ == '__main__':
    flag_test = False
    falg_gumbel_test = False
    flag_plot_gumbel_curve = False
    flag_fit_gumbel_curve_matrix = True
    flag_rainfall_divid_pro = False
    flag_rainfall_divid_pro_matrixVersion = False
    flag_rainfall_dividing = True


    # Stept one ：Generating regression curves for multi-year rainfall and determining the extreme rainfall amount for
    # n hours under 50-year, 100-year, and 200-year return periods.
    if flag_plot_gumbel_curve:
        Dr = 24 # rainfall duration
        df_test = pd.read_excel("input/input_rainfallevents/max_24hr_pre.xlsx")
        list_years = list(df_test["year"])  # max_daily_pr
        list_max_daily_pr = list(df_test["max_daily_pr"])
        list_max_rainfall_asc = np.array(list_max_daily_pr)
        # calculate eta η
        b1_sup_mao = calculate_b1_sup_mao(list_max_rainfall_asc)
        b0_sup_mao =calculate_b0_sup_mao(list_max_rainfall_asc)
        eta_sup_mao = calculate_eta_sup_mao(b0_sup_mao, b1_sup_mao)
        print(eta_sup_mao)
        gamma = sympy.S.EulerGamma.n(10)
        lambda_sup_mao = calculate_lambda_sup_mao(eta_sup_mao, b0_sup_mao, gamma)
        print(lambda_sup_mao)

        list_x = np.linspace(1.0, 201.0,100,dtype="float")
        F_sub_GUM = model_gumbel_cum_dis(list_x, lambda_sup_mao, eta_sup_mao)
        list_T = calculate_T_return_period(F_sub_GUM)
        # plot line
        plot.plot_line_simple(list_x, list_T)
        df_output = pd.DataFrame(np.array([list_x, list_T]).transpose(),columns=["cal_accu_pre","T"])
        df_output.to_excel("output/rainfall_rpp/gumbel_curve_"+str(Dr)+"hrs.xlsx", index=False)

    # The parameters of the regression curve for multi-year rainfall are generated in tensor form and stored as matrices.
    # The n-hour extreme rainfall values are explicitly identified for return periods of 50 years, 100 years, and 200 years.
    if flag_fit_gumbel_curve_matrix:
        str_fix = "luding"
        FilePath_PRC = "input/output_PDIR_Now_max_nhrs_acc/"
        # 1. Read matrix file.
        list_T = [50,100,150,200]
        list_years = list(np.linspace(2001,2020, 20, dtype="int"))
        Dr = 24 # 1 3 6 12 24
        # 1.1. Projection information
        projection, transform = DCTif.get_pro_from_img(FilePath_PRC + "/matrix_max_1hrs_acc_pre_2001.tif")  #
        dem_sample = DCTif.get_array_from_img(FilePath_PRC + "/matrix_max_1hrs_acc_pre_2001.tif")  #
        rows, cols = dem_sample.shape
        years = len(list_years)
        # 2. Declare the tensor of cumulative maximum rainfall for each year.
        list_matrix_max_rainfall = np.zeros([rows, cols, years], dtype="float")
        matrix_eta_sup_mao = np.zeros([rows, cols], dtype="float")
        matrix_lambda_sup_mao = np.zeros([rows, cols], dtype="float")
        # 2.1 Declare the tensor to store extreme cumulative rainfall values for different return periods.
        maxtrx_max_acc_pre = np.zeros([rows, cols, len(list_T)], dtype="float")
        # 3.Read the tif data and populate the tensor array.
        fileDirPath = FilePath_PRC
        pre_fix = "matrix_max_"+str(Dr)+"hrs_acc_pre"
        type_str = ".tif"
        # Loop through the files in the folder and read files with the .tif extension.
        for index in range(len(list_years)):
            tem_pre_filePath = get_precFileName(pre_fix, fileDirPath, str(list_years[index]))
            list_matrix_max_rainfall[:,:,index] = DCTif.get_array_from_img(tem_pre_filePath)
        all_recycle = rows*cols
        now_recycle = 0
        # recycle to get the parameter eta η and lambda_sup_mao
        for index_rows in range(rows):
            for index_cols in range(cols):
                now_recycle = now_recycle+1
                if now_recycle%10000==0:
                    print("执行进度：",round(100*now_recycle/all_recycle,2))
                list_max_daily_pre = list(list_matrix_max_rainfall[index_rows, index_cols,:])
                # Check if the array is entirely composed of zeros.
                flag_all_zero = np.all(np.array(list_max_daily_pre)==0.0)
                if flag_all_zero:
                    continue
                list_max_daily_pre_asc = np.array(sorted(list_max_daily_pre, reverse=False))
                # Calculating parameters
                # Compute eta η
                b1_sup_mao = calculate_b1_sup_mao(list_max_daily_pre_asc)
                b0_sup_mao = calculate_b0_sup_mao(list_max_daily_pre_asc)
                eta_sup_mao = calculate_eta_sup_mao(b0_sup_mao, b1_sup_mao)
                matrix_eta_sup_mao[index_rows,index_cols] = eta_sup_mao
                # print(eta_sup_mao)
                gamma = sympy.S.EulerGamma.n(10)
                lambda_sup_mao = calculate_lambda_sup_mao(eta_sup_mao, b0_sup_mao, gamma)
                matrix_lambda_sup_mao[index_rows,index_cols] = lambda_sup_mao
                # print(lambda_sup_mao)
                # Use binary search to find the solution.
                for index_T in range(len(list_T)):
                    ideal_T = list_T[index_T] # when T is 50 years.
                    a = 0 # Search for the left boundary
                    b = 10000 # Search for the right boundary
                    tol = 0.001 # error
                    max_acc_pre = binary_search(function_gumbel_line, a, b, tol, lambda_sup_mao, eta_sup_mao, ideal_T)
                    # record cumulative maximum rainfall
                    maxtrx_max_acc_pre[index_rows,index_cols,index_T] = max_acc_pre
        # output results
        DCTif.matrix_to_tif(matrix_eta_sup_mao, projection, transform, "float", "output/para_matrix_eta_sup_mao_Dr" + str(Dr)+str_fix + ".tif")
        DCTif.matrix_to_tif(matrix_lambda_sup_mao, projection, transform, "float",
                            "output/para_matrix_lambda_sup_mao_Dr" + str(Dr) +str_fix + ".tif")
        # output the result matrix for different recycle period (T).
        for i_output in range(len(list_T)):
            DCTif.matrix_to_tif(maxtrx_max_acc_pre[:,:,i_output], projection, transform, "float",
                                "output/para_maxtrx_max_acc_pre_Dr"+str(Dr)+ "_"+str(list_T[i_output])+"years_"+ str_fix + ".tif")


    #  (1D) Stept two：Generating rainfall amount under random rainfall pattern and decomposing the accumulated rainfall amount
    # for n hours at different time periods into rainfall sequences with finer temporal resolution.
    if flag_rainfall_divid_pro:
        # Setting input parameters
        df_test = pd.read_excel("input/input_rainfallevents/result_list_pre_1hr.xlsx")
        # List_date = list(df_test["Times"])  # max_daily_pr
        list_hr_pr = np.array(list(df_test["pre_1hr"]))
        # Parameter a0 and parameter H
        list_times_reso = [1,3,6]
        list_a = []
        # Estimation of a under different time resolution
        for index_times in range(len(list_times_reso)):
            hrs = list_times_reso[index_times]
            list_nhr_pr = np.zeros(int(len(list_hr_pr) / hrs))
            for index in range(hrs):
                list_nhr_pr = list_nhr_pr + list_hr_pr[index:len(list_hr_pr):hrs]
            W_list_nhr = generate_W_list(list_nhr_pr)
            a_nhr = estimate_a_with_W(W_list_nhr)
            print(hrs, "小时分辨率下的参数a：", a_nhr)
            list_a.append(a_nhr)
        # Calibrating parameters a0 and H based on the sample point data.
        xdata = np.array(list_times_reso)
        ydata = np.array(list_a)
        # Here you give the initial parameters for p0 which Python then iterates over
        # to find the best fit
        popt, pcov = curve_fit(func, xdata, ydata, p0=(5.0, 0.5))
        print(popt)  # This contains your two best fit parameters
        # Performing sum of squares
        p1 = popt[0] # a0
        p2 = popt[1] # H
        residuals = ydata - func(xdata, p1, p2)
        fres = sum(residuals ** 2)
        print(fres)
        xaxis = np.linspace(1.0, 25, 1000)  # we can plot with xdata, but fit will not look good
        curve_y = func(xaxis, p1, p2)
        plt.plot(xdata, ydata, '*')
        plt.plot(xaxis, curve_y, '-')
        plt.show()
        # divide rainfall under random rainfall scenarios
        if flag_rainfall_dividing:
            # Setting the cumulative rainfall amount for Dr hours at different time periods.
            para_Q_ini = 279 # mm initial cumulative rainfall amount
            para_a0 = p1
            para_H = p2
            Dr = 24 # hrs rainfall duration
            para_int_k = 7 # division times  2^7 = 128
            # rainfall division
            list_rainfall_downscaled = generate_list_rainfall_downscaled(para_Q_ini, para_a0, para_H, Dr, para_int_k)
            # plot.plot_Histogram_frm_list(list_rainfall_downscaled,100,"PR","xx")
            # output results
            df_output = pd.DataFrame(list_rainfall_downscaled,columns=["降雨序列"])
            df_output.to_excel("output/rainfall_rpp/list_rainfall_divided_"+str(Dr)+".xlsx")

    # Stept three (2D)：Generating rainfall amounts under random rainfall patterns and decomposing the cumulative rainfall
    # matrix for Dr hours at different periods into a rainfall matrix with finer temporal resolution.
    if flag_rainfall_divid_pro_matrixVersion:
        # location information
        str_fix = "luding_"
        FilePath_PRC = "input/output_PDIR_Now_max_nhrs_acc/"
        # set the pro and tran information for tif files.
        projection, transform = DCTif.get_pro_from_img(FilePath_PRC + "/matrix_max_1hrs_acc_pre_2001.tif")  #
        dem_sample = DCTif.get_array_from_img(FilePath_PRC + "/matrix_max_1hrs_acc_pre_2001.tif")  #
        rows, cols = dem_sample.shape

        # setting input parameters
        T = 150 # recycle period, unit is years
        Dr = 24 # rainfall duration # hrs , can be set as [3 6 12 24]
        filePath_Q_ini = "output/Dr_"+str(Dr)+"/"
        Q_ini_matrix = dem_sample = DCTif.get_array_from_img(filePath_Q_ini+"para_maxtrx_max_acc_pre_Dr"+str(Dr)+"_"+str(T)+"years_luding.tif")  #
        df_test = pd.read_excel("input/input_rainfallevents/result_list_pre_1hr.xlsx")
        # list_date = list(df_test["Times"])  # max_daily_pr
        list_hr_pr = np.array(list(df_test["pre_1hr"]))
        # calculate a0 and H
        list_times_reso = [1, 3, 6]
        list_a = []
        # Estimation of a under different time resolution
        for index_times in range(len(list_times_reso)):
            hrs = list_times_reso[index_times]
            list_nhr_pr = np.zeros(int(len(list_hr_pr) / hrs))
            for index in range(hrs):
                list_nhr_pr = list_nhr_pr + list_hr_pr[index:len(list_hr_pr):hrs]
            W_list_nhr = generate_W_list(list_nhr_pr)
            a_nhr = estimate_a_with_W(W_list_nhr)
            print(hrs, "小时分辨率下的参数a：", a_nhr)
            list_a.append(a_nhr)
        # Calibrating parameters a0 and H based on the sample point data.
        xdata = np.array(list_times_reso)
        ydata = np.array(list_a)
        # Here you give the initial parameters for p0 which Python then iterates over
        # to find the best fit
        popt, pcov = curve_fit(func, xdata, ydata, p0=(5.0, 0.5))
        print(popt)  # This contains your two best fit parameters
        # Performing sum of squares
        p1 = popt[0]  # a0
        p2 = popt[1]  # H
        residuals = ydata - func(xdata, p1, p2)
        fres = sum(residuals ** 2)
        print(fres)
        xaxis = np.linspace(1.0, 25, 1000)  # we can plot with xdata, but fit will not look good
        curve_y = func(xaxis, p1, p2)
        # plt.plot(xdata, ydata, '*')
        # plt.plot(xaxis, curve_y, '-')
        # plt.show()
        # divide rainfall under random rainfall scenarios
        if flag_rainfall_dividing:
            # Setting the cumulative rainfall amount for Dr hours at different time periods.
            # para_Q_ini = 279  # mm initial cumulative rainfall amount
            para_a0 = p1
            para_H = p2
            # Dr = 24  # hrs rainfall duration
            para_int_k = 7  # division times 2^7 = 128 [4 5 6 7]
            # recycle
            list_rainfall_matrix_downscaled = np.zeros([rows,cols,2**para_int_k], dtype="float")
            times_total = rows*cols
            n_now = 0
            for index_rows in range(rows):
                for index_cols in range(cols):
                    n_now = n_now + 1
                    if n_now %10000==0:
                        print("进度：",round(100*n_now/times_total,2))
                    para_Q_ini = Q_ini_matrix[index_rows, index_cols] # mm 初始降雨累积量
                    if para_Q_ini==0:
                        continue
                    # rainfall division
                    list_rainfall_downscaled = generate_list_rainfall_downscaled(para_Q_ini, para_a0, para_H, Dr, para_int_k)
                    list_rainfall_matrix_downscaled[index_rows,index_cols,:] = list_rainfall_downscaled
            # output results
            for i_output in range(2**para_int_k):
                DCTif.matrix_to_tif(list_rainfall_matrix_downscaled[:, :, i_output], projection, transform, "float",
                                    "output/outputluding/ouput_luding_rainfall_downscaled/pre_Dr" + str(Dr) + "_" + str(
                                        T) + "years_" + str_fix + str(i_output+1)+".tif")


    if flag_test:
        df_test = pd.read_excel("input/input_rainfallevents/样例_测试.xlsx")
        list_years = list(df_test["year"])  # max_daily_pr
        list_max_daily_pr = list(df_test["max_daily_pr"])
        list_max_rainfall_asc = np.array(list_max_daily_pr)
        # compute eta η
        b1_sup_mao = calculate_b1_sup_mao(list_max_rainfall_asc)
        b0_sup_mao = calculate_b0_sup_mao(list_max_rainfall_asc)
        eta_sup_mao = calculate_eta_sup_mao(b0_sup_mao, b1_sup_mao)
        print(eta_sup_mao)
        gamma = sympy.S.EulerGamma.n(10)
        lambda_sup_mao = calculate_lambda_sup_mao(eta_sup_mao, b0_sup_mao, gamma)
        print(lambda_sup_mao)

        list_x = np.linspace(1.0,301.0,10,dtype="float")
        F_sub_GUM = model_gumbel_cum_dis(list_x, lambda_sup_mao, eta_sup_mao)
        list_T = calculate_T_return_period(F_sub_GUM)
        # plot line
        plot.plot_line_simple(list_x,list_T)

    if falg_gumbel_test:
        # xs = gumbel(loc=0, scale=1, size=20000)
        # print(np.mean(xs))
        # 0.5836924011336291

        for lam in [2, 1, 0.5]:
            xs = gumbel(scale=lam, size=20000)
            plt.hist(xs, bins=200, label=f"lambda={lam}", alpha=0.5)

        plt.legend()
        plt.show()
