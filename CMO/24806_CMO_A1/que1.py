from CMO_A1 import f1, f2, f3, f4 
import numpy as np 
import matplotlib.pyplot as plt 

SR = 24806

def firstDerivative(f, x):
    h = 1e-5
    return (f(SR, x + h) - f(SR, x)) / h

def isConvex(f, interval): # to be changed
    # a f(x1) + (1-a) f(x2) >= f(ax1 + (1-a)x2)
    a1, b1 = interval
    for x1 in np.arange(a1, b1, 0.1):
        for x2 in np.arange(a1, b1, 0.1):
            x1, x2 = round(x1, 3), round(x2, 3)
            if x1 != x2:
                fx1 = f(SR, x1)
                fx2 = f(SR, x2) 
                for a in np.arange(0,1,0.01):
                    if round(a * fx1 + (1-a) * fx2, 3) < round(f(SR, a*x1 + (1-a)*x2), 3):
                        return False
    return True

def isStrictConvex(f, interval): # to be changed
    # a f(x1) + (1-a) f(x2) >= f(ax1 + (1-a)x2)
    a1, b1 = interval
    itera = 0

    x1s = np.linspace(a1, b1, 50)
    x2s = np.linspace(a1, b1, 50)

    for x1 in np.arange(a1,b1,0.1):
        for x2 in np.arange(a1, b1, 0.1):
            # x1, x2 = round(x1, 3), round(x2, 3)
            if x1 != x2:
                fx1 = f(SR, x1)
                fx2 = f(SR, x2) 
                for a in np.arange(0.01,0.99,0.01):
                    if round(a * fx1 + (1-a) * fx2, 6) <= round(f(SR, a*x1 + (1-a)*x2), 6):
                        return False
    return True

def find_min(f, alpha):
    x = 2
    epsilon = 1e-3
    optimal_fx = None
    for i in range(1, int(1e6) + 1):
        fx, gradfx = f(SR, x), firstDerivative(f, x)
        grad_norm = np.linalg.norm(gradfx)
        if  grad_norm <= epsilon:
            optimal_fx = fx 
            break 
        x = x - alpha * gradfx
    optimal_fx = fx
    return round(x, 3), round(optimal_fx, 3)

def findRoots(f):
    roots = []
    for i in np.arange(-10, 10, 0.001):
        fx = f(SR, i)
        if abs(fx) <= 1e-3:
            if roots:
                if abs(i - roots[-1]) > 1e-2:
                    roots.append(i)
            else:
                roots.append(i)

    return roots

def findMINMAX(f):
    minima = []
    maxima = []
    for i in np.arange(-10,10, 0.001):
        pmin = True
        pmax = True
        for j in np.arange(i - 0.01, i + 0.01, 0.001):
            fi = f(SR, i)
            fj = f(SR, j)
            if fi > fj:
                pmin = False
            elif fi < fj:
                pmax = False
        if pmin:
            minima.append(i)
        if pmax:
            maxima.append(i)
    return minima, maxima

def FindStationaryPoints(functionname):
    st_pts = {}

    roots = findRoots(functionname)
    minima, maxima = findMINMAX(functionname)

    st_pts['roots'] = [round(i, 3) for i in roots] 
    st_pts['minima'] = [round(i,3) for i in minima] 
    st_pts['maxima'] = [round(i,3) for i in maxima]

    return st_pts

if __name__ == "__main__":
    f1_convex = isConvex(f1, [-2,2])
    f2_convex = isConvex(f2, [-2,2])

    f1_strict = isStrictConvex(f1, [-2,2])
    f2_strict = isStrictConvex(f2, [-2,2])

    print(f'convex, f1: {f1_convex}, f2: {f2_convex}\nstrict convex: f1: {f1_strict}, f2: {f2_strict}')

    f1_min = find_min(f1, 1e-4)
    f2_min = find_min(f2, 1e-4)

    print(f"f1: x* = {f1_min[0]}, f(x*) = {f1_min[1]}")
    print(f"f2: x* = {f2_min[0]}, f(x*) = {f2_min[1]}")

    stationary_points = FindStationaryPoints(f3)

    for key, value in stationary_points.items():
        print(f"f3_{key}: {value}")