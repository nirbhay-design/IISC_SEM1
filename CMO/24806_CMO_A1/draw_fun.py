from CMO_A1 import f1, f2, f3, f4 
import numpy as np 
import matplotlib.pyplot as plt 

SR = 24806

def plot_fun(f, interval):
    x1, x2 = interval 
    # x1, x2 = int(x1), int(x2)
    x = [i for i in np.arange(x1, x2, 0.001)]
    y = []
    for i in x:
        y.append(f(SR, i))
    
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    plot_fun(f1, interval = (-0.45,-0.2))
    plot_fun(f2, interval = (-4,4))
    plot_fun(f3, interval = (-4,4))