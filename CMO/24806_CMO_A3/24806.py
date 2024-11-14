import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import os
import warnings; warnings.filterwarnings("ignore")

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

def plot_points(data, label, w, b, ac):
    e,f,g,h = 1,1,1,1
    for i, l in enumerate(label):
        if i in ac:
            if l == 1:
                plt.scatter(data[i,0], data[i,1], label='y=1' if e else None, color='r')
                e = 0
            else:
                plt.scatter(data[i,0], data[i,1], label='y=-1' if f else None, marker='s', color='r')
                f = 0
        else:
            if l == 1:
                plt.scatter(data[i,0], data[i,1], label='y=1' if g else None, color='b')
                g = 0
            else:
                plt.scatter(data[i,0], data[i,1], label='y=-1' if h else None, marker='s', color='g')
                h = 0
    x1 = np.arange(-3,3,0.1)
    x2 = -(b + w[0] * x1) / w[1]
    plt.plot(x1,x2, label='w^Tx + b', c = 'black')      
    plt.xlabel("x1")
    plt.ylabel("x2")      
    plt.legend()
    plt.show()
    
def dual_solve(data, labels):
    A = (np.outer(labels, labels)) * (np.dot(data, data.T))
    N = data.shape[0]
    lambd = cp.Variable(N)
    obj = cp.Maximize(cp.sum(lambd) - 0.5 * cp.quad_form(lambd, A, assume_PSD=True))
    constraints = [
        lambd >= 0,
        cp.sum(labels * lambd) == 0
    ]

    problem = cp.Problem(obj, constraints)
    problem.solve()

    lambda_opt = lambd.value
    lambda_opt[lambda_opt <= 1e-4] = 0

    w = np.sum((lambda_opt * labels).reshape(-1,1) * data, axis=0)
    b = labels[0] - np.dot(w, data[0])

    dual_obj = np.sum(lambda_opt) - 0.5 * np.sum(lambda_opt.reshape(-1,1).T @ A @ lambda_opt.reshape(-1,1))

    return w, b, lambda_opt, dual_obj

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
    return w.value, b.value

def que1():
    # AA = np.array([[2,-4,2,-14], [-1,2,-2,11],[-1,2,-1,7]])
    # bb = np.array([10, -6, -5]).reshape(-1,1)
    A = np.array([[1,-2,1,-7],[0,0,1,-4]])
    b = np.array([5,1]).reshape(-1,1)

    x = min_norm_soln(A, b).reshape(-1)

    print(f"least norm solution: x* = {x}, with norm: {np.linalg.norm(x):.5f}")
    # print(np.linalg.norm(AA @ x - bb))

    outs = {}
    alphas = [0.09, 0.01, 0.05, 0.1, 0.2]
    for alpha in alphas:
        x_alpha, outs[alpha] = proj_grad_descent(A, b, x, alpha)
        print(f"x* at alpha = {alpha}: {x_alpha.reshape(-1)}")
    plot(outs)

if __name__ == "__main__":
    ans = int(input("which question ans do you want [1/2]: "))
    
    if ans == 1:
        que1()
    else:
        data, label = read_csv('Data.csv', "Labels.csv")

        w, b = primal_solve(data, label)
        print(f"PRIMAL: w = {w}, b = {b}, objective value = {0.5 * np.linalg.norm(w)**2:.4f}")

        wd, bd, lambd, dual_obj = dual_solve(data, label)
        print(f"DUAL: w = {wd}, b = {bd}, objective value = {dual_obj:.4f}")

        sum_lambd = [0, 0]
        for i in range(len(lambd)):
            if label[i] == 1:
                sum_lambd[0] += lambd[i]
            else:
                sum_lambd[1] += lambd[i]
        print(f"sum_lambda (y = 1): {sum_lambd[0]:.4f}, (y=-1): {sum_lambd[1]:.4f}")

        active_constr = np.where(lambd > 0)[0]
        print(f"active constraints are: {active_constr + 1}")

        plot_points(data, label, wd, bd, active_constr)


    