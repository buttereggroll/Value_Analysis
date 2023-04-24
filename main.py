# import numpy as np
# import findroot
# import pandas as pd

# x = 1.5
# phi = [findroot.f1, findroot.f2, findroot.f3, findroot.f4, findroot.f5]
# Iter1 = {}
# Iter2 = {}
# Iter3 = {}
# Iter4 = {}
# Iter5 = {}
# Result1 = {}
# Result1 = {}
# Result1 = {}
# Result1 = {}
# Result1 = {}
# for i in range(5):
#     result[i], iter[i] = findroot.aitken_method(x, phi[i])
#     print(len(result[i]))
#
#
#
# Output = np.zeros((max(iter.values()) + 1, 6))
# for key in result.keys():
#     Input = result[key]
#     for index in range(len(Input)):
#         Output[index, key + 1] = Input[index]
#
# Output[:, 0] = range(0, Output.shape[0])
# Output = np.where(Output == 0, " ", Output)
# Output = pd.DataFrame(Output, columns=['迭代次数', 'phi1', 'phi2', 'phi3', 'phi4', 'phi5'])
# #Output['迭代次数'] = Output['迭代次数'].astype(np.float32).astype(str)
# Output.to_csv("aitken_method.csv", index=False, encoding='gbk')
#


# findroot.bisection(1.3, 1.5)
# findroot.aitken_method(1.5, findroot.f5)
# findroot.static_secant_method(1.3, 1.5, findroot.f)
# findroot.dynamic_secant_method(1.3, 1.5, findroot.f)


# x = 1.5
# phi = [findroot.f1, findroot.f2, findroot.f3, findroot.f4, findroot.f5]
# iter = {}
# result = {}
#
# result[0], iter[0] = findroot.newton_method(x, findroot.f, findroot.f_derivative)
#
# Output = np.zeros((max(iter.values()) + 1, 2))
# for key in result.keys():
#     Input = result[key]
#     for index in range(len(Input)):
#         Output[index, key + 1] = Input[index]
#
# Output[:, 0] = range(0, Output.shape[0])
# Output = np.where(Output == 0, " ", Output)
# Output = pd.DataFrame(Output, columns=['迭代次数', '迭代值'])
# #Output['迭代次数'] = Output['迭代次数'].astype(np.float32).astype(str)
# Output.to_csv("newton_method.csv", index=False, encoding='gbk')


# x = 1.5
# phi = [findroot.f1, findroot.f2, findroot.f3, findroot.f4, findroot.f5]
# Iter = {}
# Result = {}
#
# # 定端点为x0=1.3
# Result[0], Iter[0] = findroot.static_secant_method(x, 1.3, findroot.f)
#
# Output = np.zeros((max(Iter.values()) + 1, 2))
# for key in Result.keys():
#     Input = Result[key]
#     for index in range(len(Input)):
#         Output[index, key + 1] = Input[index]
#
# Output[:, 0] = range(0, Output.shape[0])
# Output = np.where(Output == 0, " ", Output)
# Output = pd.DataFrame(Output, columns=['迭代次数', '迭代值'])
# Output.to_csv("static_secant_method.csv", index=False, encoding='gbk')


# x = 1.5
# phi = [findroot.f1, findroot.f2, findroot.f3, findroot.f4, findroot.f5]
# Iter = {}
# Result = {}
#
# # 初始端点为x0=1.3
# Result[0], Iter[0] = findroot.dynamic_secant_method(x, 1.3, findroot.f)
#
# Output = np.zeros((max(Iter.values()) + 2, 2))
# for key in Result.keys():
#     Input = Result[key]
#     for index in range(len(Input)):
#         Output[index, key + 1] = Input[index]
#
# Output[:, 0] = range(0, Output.shape[0])
# Output = np.where(Output == 0, " ", Output)
# Output = pd.DataFrame(Output, columns=['迭代次数', '迭代值'])
# Output.to_csv("dynamic_secant_method.csv", index=False, encoding='gbk')

# Matrix1 = np.array([[2.51, 1.48, 4.53, 0.05], [1.48, 0.93, -1.3, 1.03], [2.68, 3.04, -1.48, -0.53]])
# dic1 = gauss_method(Matrix1)
#
# Matrix1 = np.array([[2.51, 1.48, 4.53, 0.05], [1.48, 0.93, -1.3, 1.03], [2.68, 3.04, -1.48, -0.53]])
# dic2 = gauss_column_pivoting(Matrix1)
#
# print('高斯消元法得到的解集：', list(dic1[i] for i in range(1, len(Matrix1[:, 0]) + 1)))
# print('列主元高斯消元法得到的解集：', list(dic2[i] for i in range(1, len(Matrix1[:, 0]) + 1)))
# list1 = [1.457, -1.595, -0.275]
# list2 = [1.454, -1.59, -0.275]
#
# precise_solve = [1.4531, -1.589195, -0.2748947]
# error1 = 0
# error2 = 0
# for i in range(3):
#     error1 += abs(list1[i] - precise_solve[i])
#     error2 += abs(list2[i] - precise_solve[i])
#
# print('高斯消元法：{:.17f}'.format(error1 / 3))
# print('列主元高斯消去法：{:.17f}'.format(error2 / 3))


# dic = gauss_jordan_method(Matrix1)
# print('列主元高斯消元法得到的解集：')
# for i in range(1, len(Matrix1[:, 0]) + 1):
#     print('x{}为{}'.format(i, dic[i]))
# print(get_lower_triangular_matrix(Matrix1))
# b = np.array([0.05, 1.03, -0.53])
# print(get_lower_triangular_matrix(Matrix1), '\n')


# Matrix1 = np.array([[1.0, 2, 1, -2, 4], [2, 5, 3, -2, 7], [-2, -2, 3, 5, -1], [1, 3, 2, 3, 0]])
#
# dic1 = gauss_jordan_method(Matrix1)
# print('\n gauss-jordan方法得到的解集：')
# for i in range(1, len(Matrix1[:, 0]) + 1):
#     print('x{}为{}'.format(i, dic1[i]))
#
# dic2 = doolittle_method(Matrix1)
# print('\n gauss-jordan方法得到的解集：')
# for i in range(1, len(Matrix1[:, 0]) + 1):
#     print('x{}为{}'.format(i, dic2[i]))


# import math
#
# print("计算一元二次方程的根")
# a = int(input("请输入二次项系数："))
# b = int(input("请输入一次项系数："))
# c = int(input("请输入常数项系数："))
#
# def solveX1(a, b, c):
#     while b*b < 4*a*c:
#         print("该方程无根,请重新输入系数")
#         a = int(input("请输入二次项系数："))
#         b = int(input("请输入一次项系数："))
#         c = int(input("请输入常数项系数："))
#     return 4.0 * a * c / ((-1.0 * b + math.sqrt(b * b - 4.0 * a * c)) * 2 * a)
#
#
# def solveX2(a, b, c):
#     while b*b < 4*a*c:
#         print("该方程无根,请重新输入系数")
#         a = int(input("请输入二次项系数："))
#         b = int(input("请输入一次项系数："))
#         c = int(input("请输入常数项系数："))
#     return 4.0 * a * c / ((-1.0 * b - math.sqrt(b * b - 4.0 * a * c)) * 2 * a)
#
# x1 = solveX1(a, b, c)
# x2 = solveX2(a, b, c)
#
# print("左根为{},右根为{}".format(x1, x2))

# import math
#
# x = float(input("输入需要求正平方根的值："))
#
# print("sqrt(", x, ")= ", math.sqrt(x))
#
# result1 = math.sqrt(x+1) - math.sqrt(x)
#
# print("普通计算方法", result1)
#
# result2 = 1 / (math.sqrt(x+1) + math.sqrt(x))
#
# print("变换公式方法", result2)

# x = 0.0
#
# for i in range(1000000):
#     x = x + 0.1
#
# print("sum result = ", x)

# x = 1e10
# y = 1e-8
#
# for i in range(10000000):
#     x = x + y
#
# print(x)

# from decimal import *
#
# x = Decimal("0.0")
#
# for i in range(1000000):
#     x = x + Decimal("0.1")
#
# print("sum result = ", x)

# from fractions import Fraction
#
# print(Fraction(5, 10), Fraction(3, 15))
#
# print(Fraction(1, 3) + Fraction(1, 7))
#
# print(Fraction(5, 3) * Fraction(6, 7) * Fraction(3, 2))

# import numpy as np
#
# def f(x):
#     return x**4 + 2*x**2 - 3
#
# #函数用法：输入要计算零点的函数的左右两点，以及要求的精度LIMIT，还有要写入的文件（目前仅测试通过txt文件）
#

# xlow = -2932
# xupp = 1000
# LIMIT = 1e-15
#
# with open("./record.txt", 'w', encoding='utf-8') as f1:
#     bisection(xlow, xupp, LIMIT, f1)

# x1 = x2 = x3 = x4 = x5 = 1.5
#
# for i in range(10):
#     if i < 4:
#         x1 = f1(x1)
#     x2 = f2(x2)
#     x3 = f3(x3)
#     x4 = f4(x4)
#     x5 = f5(x5)
#     if i < 5:
#         print("x1 = {}, x2 = {:.5f}, x3 = {:.5f} x4 = {:.5f} x5 = {:.5f}".format(x1, x2, x3, x4, x5))
#     else:
#         print("x2 = {:.5f}, x3 = {:.5f} x4 = {:.5f} x5 = {:.5f}".format(x2, x3, x4, x5))

# Matrix1 = np.array([[2.51, 1.48, 4.53, 0.05], [1.48, 0.93, -1.3, 1.03], [2.68, 3.04, -1.48, -0.53]])
# print(matrix_norm_1(Matrix1))
# print(matrix_norm_infinite(Matrix1))

# Matrix1 = np.array([[2.51, 1.48, 4.53], [1.48, 0.93, -1.3], [2.68, 3.04, -1.48]])
# print(np.linalg.eigvals(Matrix1))
# eigenvalue_list = np.linalg.eigvals(Matrix1)
# print(abs(eigenvalue_list[1]))
# print(np.sqrt(1.55835786 ** 2 + 0.50842941 ** 2))
# print(matrix_norm_1(Matrix1))
# print(np.linalg.norm(Matrix1, ord=1))
# print(matrix_norm_infinite(Matrix1))
# print(np.linalg.norm(Matrix1, ord=np.inf))
# print(matrix_norm_spectral(Matrix1))
# print(np.linalg.norm(Matrix1, ord=2))


# Matrix1 = np.array([[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]])
# print(matrix_norm_infinite(Matrix1))
# print(matrix_norm_1(Matrix1))
# print(matrix_norm_spectral(Matrix1))
# print(cond(Matrix1))

# Matrix1 = np.array([[10, 7, 8, 7], [7, 5, 6, 5], [8, 6, 10, 9], [7, 5, 9, 10]])
# b = np.array(np.transpose([32, 23, 33, 31]))
# Matrix1 = np.array([[11, -3, -2, 3], [-1, 5, -3, 6], [-2, -12, 19, -7], [7, 5, 9, 10]])
# b = np.array(np.transpose([32, 23, 33, 31]))
# Matrix1 = np.array([[10, 3, 1], [2, -10, 3], [1, 3, 10]])
# b = np.array(np.transpose([14, -5, 14]))
# # Matrix1 = np.array([[2.51, 1.48, 4.53], [1.48, 0.93, -1.3], [2.68, 3.04, -1.48]])
# # b = np.array(np.transpose([0.05, 1.03, -0.53]))
# # b_perturb = np.array(np.transpose([32.1, 22.9, 33.1, 30.9]))
# dic = doolittle_method(get_merge_matrix(Matrix1, b))
# # dic_perturb = doolittle_method(get_merge_matrix(Matrix1, b_perturb))
# # print(dic)
# # print(turn_to_list(dic))
# matrix = get_merge_matrix(Matrix1, b)
# dic2 = gauss_method(matrix)
# iter_matrix, iter_array = get_iter_matrix(get_merge_matrix(Matrix1, b))
# # print(matrix)
# print(iter_matrix, iter_array)
# # print(b[3])
# # print(jacobi_iter(iter_matrix, iter_array, [0, 0, 0, 0]))
# x1_list = turn_to_list(dic)
# x2_list = turn_to_list(dic2)
# print(jacobi_method(matrix))
# x3_list = jacobi_method(matrix)
# print(turn_to_list(dic2))
# print(turn_to_list(dic))
# for i in range(len(dic.keys())):
#     print(np.dot(x1_list, Matrix1[i, :]) - b[i])
# for i in range(len(dic2.keys())):
#     print(np.dot(x2_list, Matrix1[i, :]) - b[i])
# for i in range(len(dic2.keys())):
#     print(np.dot(x3_list, Matrix1[i, :]) - b[i])


# print(jacobi_iter(iter_matrix, iter_array, np.array([0, 0, 0])))

# x = list(dic.values())
# x.reverse()
# x_perturb = list(dic_perturb.values())
# x_perturb.reverse()
# x = np.array(x)
# x_perturb = np.array(x_perturb)
# print(x)
# print(gauss_seidel(Matrix1))
# print(l)
# print(l_perturb)
# print(np.linalg.norm(l_perturb - l, np.inf) / np.linalg.norm(l, np.inf) )
# print(relative_error(x, x_perturb, np.inf))
# print(relative_error(x, x_perturb, 1))
# print(relative_error(b, b_perturb, np.inf))
# print(relative_error(b, b_perturb, 1))
# print(np.linalg.det(Matrix1))
# print(np.linalg.cond(Matrix1))
# print(np.linalg.eigvals(Matrix1))
# print(np.linalg.norm())

# Matrix1 = np.array([[10, 3, 1], [2, -10, 3], [1, 3, 10]])
# b = np.array(np.transpose([14, -5, 14]))
# # jacobi_method(get_merge_matrix(Matrix1, b))
# iter_matrix, iter_array = get_iter_matrix(get_merge_matrix(Matrix1, b))
# # print(iter_matrix, iter_array)
# # print(gauss_seidel_iter(iter_matrix, iter_array, [0, 0, 0]))
# print(gauss_seidel(get_merge_matrix(Matrix1, b)))

# Matrix1 = np.array([[1, 2, -2, 0], [1, 1, 1, 0], [2, 2, 1, 0]])
# Matrix2 = np.array([[2, -1, 1, 0], [1, 1, 1, 0], [1, 1, -2, 0]])
# iter_matrix1, iter_array1 = get_iter_matrix(Matrix1)
# iter_matrix2, iter_array2 = get_iter_matrix(Matrix2)
#
# print('矩阵（1）的谱半径为：{}'.format(spectral_radius(iter_matrix1)))
# print('矩阵（2）的谱半径为：{}'.format(spectral_radius(iter_matrix2)))
# # print(matrix_norm_1(iter_matrix1))
# # print(matrix_norm_1(iter_matrix2))
# # print(matrix_norm_infinite(Matrix1))
# # print(matrix_norm_infinite(Matrix2))
# matrix = np.array([[11, -3, -2, 3], [-1, 5, -3, 6], [-2, -12, 19, -7]])
# jacobi_method(matrix, max_iters=3)
# gauss_seidel(matrix, max_iters=3)
#

# gauss_solution = []
# gauss_jordan_solution = []
# doolittle_solution = []
#
# for i in range(1, 10):
#     b = derive_b(get_hilbert(i), np.ones((i, 1)))
#     hilbert = get_hilbert(i)
#     matrix = get_merge_matrix(hilbert, b)
#     gauss_solution = turn_to_list(gauss_method(matrix))
#     gauss_jordan_solution = turn_to_list(gauss_jordan_method(matrix))
#     doolittle_solution = turn_to_list(doolittle_method(matrix))
#     print('第{}阶希尔伯特矩阵，解为全1\n'.format(i), gauss_solution, '\n')
#     print(gauss_jordan_solution, '\n')
#     print(doolittle_solution, '\n\n')
#
# for i in range(1, 10):
#     b = derive_b(get_hilbert(i), np.ones((i, 1)))
#     hilbert = get_hilbert(i)
#     matrix = get_merge_matrix(hilbert, b)
#     jacobi_solution = jacobi_method(matrix)
#     gauss_seidel_solution = gauss_seidel(matrix)
#     print('第{}阶希尔伯特矩阵，解为全1'.format(i))
#     print(jacobi_solution)
#     print(gauss_seidel_solution, '\n')
#
# for i in range(1, 10):
#     print('第{}阶希尔伯特矩阵:'.format(i), np.linalg.eigvals(get_hilbert(i)), '\n')
#
# for i in range(1, 10):
#     b = derive_b(get_hilbert(i), np.ones((i, 1)))
#     hilbert = get_hilbert(i)
#     matrix = get_merge_matrix(hilbert, b)
#     m, b = get_iter_matrix(matrix)
#     print('第{}阶希尔伯特矩阵:'.format(i), spectral_radius(m), '\n')

from interpolation import *


# # 线性插值
# data = [[0, 0], [1, 2]]
# print(Lg(data, 1.5))
# Lg_plot(data, 100)
#
# print(NT(data, 1.5))
# NT_plot(data, 100)
#
# print(spline(data, 1.5))
# spline_plot(data, 100)
#
# print(DivideLine(data, 1.5))
# Divline_plot(data, 100)
#
# # 二次多项式插值
# data = [[0, 0], [1, 2], [2, 3]]
# print(Lg(data, 1.5))
# Lg_plot(data, 100)
#
# print(NT(data, 1.5))
# NT_plot(data, 100)
#
# print(spline(data, 1.5))
# spline_plot(data, 100)
#
# print(DivideLine(data, 1.5))
# Divline_plot(data, 100)
#
# # 四次多项式插值
# data = [[0, 0], [1, 2], [2, 3], [3, 8]]
# print(Lg(data, 1.5))
# Lg_plot(data, 100)
#
# print(NT(data, 1.5))
# NT_plot(data, 100)
#
# print(spline(data, 1.5))
# spline_plot(data, 100)
#
# print(DivideLine(data, 1.5))
# Divline_plot(data, 100)
#
#
# # 五次多项式插值
# data = [[0, 0], [1, 2], [2, 3], [3, 8], [4, 2]]
# print(Lg(data, 1.5))
# Lg_plot(data, 100)
#
# print(NT(data, 1.5))
# NT_plot(data, 100)
#
# print(spline(data, 1.5))
# spline_plot(data, 100)
#
# print(DivideLine(data, 1.5))
# Divline_plot(data, 100)
#
#
# # 六次多项式插值
# data = [[0, 0], [1, 2], [2, 3], [3, 8], [4, 2], [5, 7]]
# print(Lg(data, 1.5))
# Lg_plot(data, 100)
#
# print(NT(data, 1.5))
# NT_plot(data, 100)
#
# print(spline(data, 1.5))
# spline_plot(data, 100)
#
# print(DivideLine(data, 1.5))
# Divline_plot(data, 100)
#
#
# # 七次多项式插值
# data = [[0, 0], [1, 2], [2, 3], [3, 8], [4, 2], [5, 7], [6, 8]]
# print('\n', Lg(data, 1.5))
# Lg_plot(data, 100)
#
# print(NT(data, 1.5))
# NT_plot(data, 100)
#
# print(spline(data, 1.5))
# spline_plot(data, 1000)
#
# print(DivideLine(data, 1.5))
# Divline_plot(data, 100)

# 实验数据插值
data1 = [[-5.0, -0.1923], [-4.5, -0.2118], [-4.0, -0.2353], [-3.5, -0.2642], [-3.0, -0.3], [-2.5, -0.3448],
         [-2.0, -0.4000], [-1.5, 0.4615], [-1.0, -0.5000], [-0.5, -0.4000], [0, 0], [0.5, 0.4000], [1.0, 0.5000],
         [1.5, 0.4615], [2.0, 0.4000], [2.5, 0.3448], [3.0, 0.3000], [3.5, 0.2642], [4.0, 0.2353], [4.5, 0.2118],
         [5.0, 0.1923]]
data2 = [[-5.0, 0.0016], [-4.5, 0.002], [-4.0, 0.0025], [-3.5, 0.0033], [-3.0, 0.0044], [-2.5, 0.0064], [-2.0, 0.0099],
         [-1.5, 0.0175], [-1.0, 0.0385], [-0.5, 0.1379], [0, 1.0000], [0.5, 0.1379], [1.0, 0.0385], [1.5, 0.0175],
         [2.0, 0.0099], [2.5, 0.0064], [3.0, 0.0044], [3.5, 0.0033], [4.0, 0.0025], [4.5, 0.0020], [5.0, 0.0016]]
print(Lg(data1, 1.3))
Lg_plot(data1, 100)
Lg_plot(data2, 100)

print(NT(data1, 1.3))
NT_plot(data1, 100)
NT_plot(data2, 100)

print(spline(data1, 1.3))
spline_plot(data1, 1000)
spline_plot(data2, 1000)

print(DivideLine(data1, 1.3))
Divline_plot(data1, 100)
Divline_plot(data2, 100)
