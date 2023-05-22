def euler_method(yi, h, f, t):
    """欧拉方法，返回第i+1个y的值"""
    return yi + h * f(t, yi)

def heun_method(yi, h, f, t, limit=1e-5):
    """改进的欧拉方法（休恩方法），返回第i+1个y的值，默认最大误差限为1e-5"""
    iter_y_list = [yi + h * f(t, yi)]   # 依次储存第i+1个y第k次迭代的值
    while True:
        iter_y_list.append(yi + h * (f(t, yi) + f(t+h, iter_y_list[-1])) / 2)
        if iter_y_list[-1] - iter_y_list[-2] <= limit:
            break
    return iter_y_list[-1]
