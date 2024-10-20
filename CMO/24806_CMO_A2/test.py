from oracles import f1, f2, f3 
import numpy as np 
sr = 24806

def plotter(x,y):
    import matplotlib.pyplot as plt
    plt.plot(x,y)
    plt.show()
srno = 24806
def quasi_newton_sr1(x0):
    x = x0
    G = np.eye(len(x0))
    u = -np.dot(G, f2(x, srno, 1))
    xvals=[]
    xvals.append(x)
    yvals=[]
    yvals.append(f2(np.array(x), srno, 0))
    fb = f2(np.zeros(len(x0)), srno, 1)
    
    for i in range(100):
        g1 = f2(x, srno, 1)
        alpha = -np.dot(g1, u) / (2* (f2(u, srno, 0)) - np.dot(fb, u))
        deli = alpha * u
        x = x + alpha * u
        g2 = f2(x, srno, 1)
        gamma = g2 - g1
        
        if np.dot(deli - np.dot(G, gamma).T, gamma) != 0:
            G = G + np.outer(deli - np.dot(G, gamma), deli - np.dot(G, gamma).T) / np.dot(deli - np.dot(G, gamma).T, gamma)
        
        u = -np.dot(G, f2(x, srno, 1))
        deli = alpha * u
        xvals.append(x)
        yvals.append(f2(x, srno, 0))
    print(np.round(x,4),np.round(G,4), np.round(u,4), np.round(deli,4))
    plotter(list(range(101)),yvals)
    return [round(i,4) for i in x], yvals,G

if __name__ == "__main__":
    # a = np.zeros(5)
    # fx = f2(a, sr, 0)
    # print(fx)

    # b = f2(a, sr, 1)
    # print(b)

    # x = np.ones(5)
    # hfx1 = f2(x, sr, 0) - np.dot(b, x) 
    # hfx2 = f2(-x, sr, 0) - np.dot(b, -x) 

    # print(hfx1)
    # print(hfx2)

    quasi_newton_sr1(np.zeros(5))

