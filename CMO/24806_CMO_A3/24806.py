import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import cvxpy as cp

def projection(A, b, z):
    # z and b both are (n, 1)
    return z + np.dot((A.T @ np.linalg.inv(A @ A.T)), (b - np.dot(A,z)))

def min_norm_soln(A, b):
    z = np.zeros((A.shape[1], 1))
    return projection(A, b, z)

def proj_grad_descent(A,b,optx,alpha):
    x = np.ones((A.shape[1], 1))
    norm_diff = []
    for i in range(200):
        z = (1 - alpha) * x
        x = projection(A, b, z)
        norm_diff.append(np.linalg.norm(x - optx))
    return x, norm_diff

def plot(outs):
    for alpha, values in outs.items():
        plt.plot(values, label=f'{alpha}')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("||x - x*||")
    plt.show()

def read_csv(data, label):
    with open(data, 'r') as f:
        d = f.readlines()
        d = list(map(lambda x: [float(i) for i in x.replace("\n", "").split(",")], d))

    with open(label, 'r') as f:
        l = f.readlines()
        l = list(map(lambda x: float(x.replace("\n", "")), l))
    return np.array(d), np.array(l)

def plot_points(data, label):
    for i, l in enumerate(label):
        if l == 1:
            plt.scatter(data[i,0], data[i,1], color='b')
        else:
            plt.scatter(data[i,0], data[i,1], marker='s', color='r')
    plt.show()
    
def dual_solve(data, labels):
    A = (np.outer(labels, labels)) * (data @ data.T)
    N = data.shape[0]

    lambd = cp.Variable(N)
    obj = cp.Maximize(cp.sum(lambd) - 0.5 * cp.quad_form(lambd, A))
    constraints = [
        lambd >= 0,
        labels @ lambd == 0
    ]

    problem = cp.Problem(obj, constraints)
    problem.solve()

    lambda_opt = lambd.value
    print(lambda_opt)

def primal_solve(data, labels):
    N = data.shape[1]
    w = cp.Variable(N)
    b = cp.Variable(1)
    objective = cp.Minimize(0.5 * cp.quad_form(w, np.eye(N)))
    constrns = [
        labels[i] * (w @ data[i] + b) >= 1 for i in range(len(labels))
    ]
    prob = cp.Problem(objective, constrns)
    prob.solve()
    print(w.value, b.value)

def que1():
    # AA = np.array([[2,-4,2,-14], [-1,2,-2,11],[-1,2,-1,7]])
    # bb = np.array([10, -6, -5]).reshape(-1,1)
    A = np.array([[1,-2,1,-7],[0,0,1,-4]])
    b = np.array([5,1]).reshape(-1,1)

    x = min_norm_soln(A, b).reshape(-1)

    print(f"least norm solution: x* = {x}, with norm: {np.linalg.norm(x)}")
    # print(np.linalg.norm(AA @ x - bb))

    outs = {}
    alphas = [0.09, 0.01, 0.05, 0.1, 0.2]
    for alpha in alphas:
        x_alpha, outs[alpha] = proj_grad_descent(A, b, x, alpha)
        print(f"x* at alpha = {alpha}: {x_alpha.reshape(-1)}")
    plot(outs)

if __name__ == "__main__":
    # que1()
    data, label = read_csv('Data.csv', "Labels.csv")

    print(data)
    print(label)

    # primal_solve(data, label)
    dual_solve(data, label)


    