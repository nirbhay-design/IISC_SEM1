import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
 
 
# def compare(x):
#     newA = np.array([[1,-2,1,7],[0,0,1,5]])
#     newA = np.array([[2 ,-4 ,2 ,-14],[-1,2,-2,11],[-1,2,-1,7]])
def get_proj(A,b,z):
    b = b.reshape(-1,1)
    z = z.reshape(-1,1)
    # print(z.shape)
    x = A.T @ LA.inv(A @ A.T) @ (b - A@z) + z
    return x
 
def f(x):
    return 0.5*(LA.norm(x)**2)
def check_conv(x):
    my_x = np.array([1,1,1,1]).reshape(-1,1)
    # print(my_x.shape,x.shape)
    atx = (x.T @ (my_x - x))
    print(f" ATx : {atx}")
    return (atx)
def projected_gradient_descent(f,A,b,initialx,alpha):
    stopping_criteria = -1
    it = 0
    curve = []
    while (stopping_criteria<0) and (it<100):
        curve.append(f(initialx))
        movex = initialx*(1-alpha)
        movex_p = get_proj(A,b,movex)
       
 
        initialx = movex_p
        print(movex_p)
        stopping_criteria = check_conv(movex_p)
        it +=1
    # print(stopping_criteria)
    return curve
 
 
 
def SLE():
    A = np.array([[1,-2,1,7],[0,0,1,5]])
    b = np.array([5,1])
    LNS = get_proj(A,b,np.array([0,0,0,0]))
    print(f"Least Norm Solution:{LNS}")
    print(f"Norm of the solution:{LA.norm(LNS)}")
 
    x0 = np.array([3,4,5,1])
    alpha = 0.2
    curve = projected_gradient_descent(f,A,b,x0,alpha)
    plt.plot(curve)
    plt.show()
 
 
    # print()
 
# SLE()
import cvxpy as cp 

def get_data(data):
    with open(data, 'r') as f:
        d = f.readlines()
        dv = []
        for i in d:
            i = i.replace("\n", " ").split(',')
            fi = [float(j) for j in i]
            dv.append(fi)
    return np.array(dv)
def get_label(label):
    with open(label, 'r') as f:
        l = f.readlines()
        lv = []
        for i in l:
            lv.append(float(i.replace("\n", "")))
    return np.array(lv)

def solve_primal(X, y):
    w = cp.Variable(X.shape[1])
    b = cp.Variable(1)
    O = cp.Minimize(cp.multiply(0.5, cp.norm(w) ** 2))
    C = []
    for i in range(len(y)):
        C.append(cp.multiply(y[i], (w @ X[i] + b)) >= 1)
    
    prob = cp.Problem(O, C)
    prob.solve()

    opt_w, opt_b = w.value, b.value

    print("optimal w:")
    print(opt_w)

    print("optimal b:")
    print(opt_b)

    print("primal optimal function value:")
    print(LA.norm(opt_w) ** 2 / 2)

def solve_dual(X, y):
    K = (y.reshape(-1,1) @ y.reshape(-1,1).T) * (X @ X.T)
    N = X.shape[0]
    l = cp.Variable(N)
    O = cp.Maximize(cp.sum(l) - cp.multiply(0.5, cp.quad_form(l, K, assume_PSD=True)))
    C = [l >= 0, cp.sum(cp.multiply(y, l)) == 0]

    problem = cp.Problem(O, C)
    problem.solve()

    l_opt = l.value
    l_opt[l_opt <= 1e-4] = 0

    w = np.sum((l_opt * y).reshape(-1,1) * X, axis=0)
    b = y[0] - np.dot(w, X[0]) # since 1 is active constraint

    DO = np.sum(l_opt) - 0.5 * np.sum(l_opt.reshape(-1,1).T @ K @ l_opt.reshape(-1,1))

    print("optimal w:")
    print(w)

    print("optimal b:")
    print(b)

    print("lambda values:")
    print(l_opt)

    print("dual optimal function value:")
    print(DO)

    SY1 = 0; SY_1 = 0
    for i, j  in enumerate(y):
        if j == 1:
            SY1 += l_opt[i]
        else:
            SY_1 += l_opt[i]
    print(f"gamma = {SY1}")
    
X, y = get_data('Data.csv'), get_label('Labels.csv')
solve_primal(X, y)
solve_dual(X, y)
