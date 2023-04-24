from basic import *


# 二分法
def bisection(xlow, xupp, LIMIT=1e-6):
    print('\n二分法:\n')
    iter = 0

    while xupp - xlow > LIMIT:
        xmid = xlow + (xupp - xlow) / 2
        iter += 1
        if np.sign(f(xmid)) * np.sign(f(xlow)) < 0:
            xupp = xmid
        else:
            xlow = xmid
        #file.write("iter:{:.15g}  xlow:{:.15g}  xupp:{:.15g},".format(iter, xlow, xupp) + '\t')
        print("迭代次数 iter = {}, 上界：{:.5f}, 下界：{:.5f}".format(iter, xlow, xupp))


# 最大迭代次数为20
def aitken_method(x, f, LIMIT=1e-6,
                  MAX_ITER=20, MAX_NUM=1e6):
    print('\naitken方法：\n')
    iter = 0
    iter_value_list = [x]
    x = f(iter_value_list[-1])
    iter_value_list.append(x)
    while True:
        iter += 1
        x = f(iter_value_list[-1])
        x = aitken_iter(iter_value_list[-2], iter_value_list[-1], x)
        iter_value_list.append(x)
        print("迭代次数 iter = {}, Aitken方法： {:.5f}".format(iter, x))
        if abs(iter_value_list[-2] - iter_value_list[-1]) <= LIMIT or iter >= MAX_ITER-1 \
        or abs(x) >= MAX_NUM:
            break
    return iter_value_list, iter


def steffensen_method(x, f, LIMIT=1e-6,
                      MAX_ITER=20):
    print('\nSteffensen方法：\n')
    iter = 0
    iter_value_list = [x]
    while f(f(x)) - 2 * f(x) + x != 0 and abs(
            steffensen_iter(x, f) - x) > LIMIT and iter < MAX_ITER:  # 用x表达Aitken方法每次迭代的值，用y表达Steffensen方法每次迭代的值
        x = steffensen_iter(x, f)
        iter_value_list.append(x)
        print("迭代次数 iter = {}, Steffensen方法： {:.5f}".format(iter, x))
        iter += 1
    return iter_value_list, iter


def newton_method(x, f, f_derivative, LIMIT=1e-6,
                  MAX_ITER=20):
    iter = 0
    iter_value_list = [x]
    print('\n牛顿切线法：\n\n迭代次数：{}, 当前值：{:.5f}'.format(iter, x))
    while abs(newton_iter(x, f, f_derivative) - x) > LIMIT and iter < MAX_ITER:
        x = newton_iter(x, f, f_derivative)
        iter_value_list.append(x)
        iter += 1
        print('迭代次数：{}, 当前值：{:.5f}'.format(iter, x))
    return iter_value_list, iter


# 最大迭代次数为20
def static_secant_method(x, x0, f, LIMIT=1e-6,
                         MAX_ITER=20):
    iter = 0
    iter_value_list = [x]
    print('\n定端点弦截法：\n\n迭代次数：{}, 当前值：{:.5f}'.format(iter, x))
    while abs(static_secant_iter(x, x0, f) - x) > LIMIT and iter < MAX_ITER:
        x = static_secant_iter(x, x0, f)
        iter_value_list.append(x)
        iter += 1
        print('迭代次数：{}, 当前值：{:.5f}'.format(iter, x))
    return iter_value_list, iter


# x为迭代值，x0为一开始选取的参考定点，f为phi(x)，最大迭代次数为20次，精度为1e-6
def dynamic_secant_method(x, x0, f, LIMIT=1e-6,
                          MAX_ITER=20):
    iter = 0
    iter_value_list = [x]
    print('\n动端点弦截法：\n\n迭代次数：{}, 当前值：{:.5f}'.format(iter, x))
    pre_x = x
    x = static_secant_iter(x, x0, f)
    iter_value_list.append(x)
    while abs(dynamic_secant_iter(x, pre_x, f) - x) > LIMIT and iter < MAX_ITER-1:
        temp = x
        x = dynamic_secant_iter(x, pre_x, f)
        iter_value_list.append(x)
        pre_x = temp
        iter += 1
        print('迭代次数：{}, 当前值：{:.5f}'.format(iter, x))
    return iter_value_list, iter




