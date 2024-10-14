from oracles import f1, f2, f3 
import numpy as np 
sr = 24806

if __name__ == "__main__":
    a = np.zeros(5)
    fx = f2(a, sr, 0)
    print(fx)

    b = f2(a, sr, 1)
    print(b)

    x = np.ones(5)
    hfx1 = f2(x, sr, 0) - np.dot(b, x) 
    hfx2 = f2(-x, sr, 0) - np.dot(b, -x) 

    print(hfx1)
    print(hfx2)

