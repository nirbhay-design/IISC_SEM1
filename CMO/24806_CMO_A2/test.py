from oracles import f1, f2, f3 
import numpy as np 
sr = 24806

if __name__ == "__main__":
    a = np.zeros(5)
    fx = f2(a, sr, 0)
    print(fx)

    gfx = f2(a, sr, 1)
    print(gfx)

    b = 2*np.ones(5)
    hfx1 = f2(b, sr, 2) 
    hfx2 = f2(-b, sr, 2) 

    print(hfx1)
    print(hfx2)