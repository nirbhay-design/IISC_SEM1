from oracles import f1, f2, f3
import numpy as np 

sr = 24806

def gradf(A, x, b):
    return np.dot(A,x) - b

def conjugate_gradient(A, b, x0):
    gs = [gradf(A, x0, b)]
    conj_grads = [-gs[0]]
    x = x0
    epsilon = 1e-4
    T = 0
    while True:
        if np.linalg.norm(gs[-1]) < epsilon :
            break
        prev_u = conj_grads[-1]
        prev_grad = gs[-1]
        denominator_coefs = np.dot(prev_u, np.dot(A, prev_u))
        alpha = - np.dot(prev_grad, prev_u) / denominator_coefs
        x = x + alpha * prev_u
        new_grad = gradf(A, x, b)
        beta = np.dot(new_grad, np.dot(A, prev_u)) / denominator_coefs
        new_u = -new_grad + beta * prev_u 
        gs.append(new_grad)
        conj_grads.append(new_u)
        T += 1
    return {
        "x":x,
        "cg": conj_grads,
        "T": T
    }

def que1():
    A, b = f1(sr, True)
    b = b.reshape(-1)
    x0 = np.array([0 for _ in range(b.shape[0])])
    out = conjugate_gradient(A, b, x0)
    print("#### PART2 ####")
    print(f"x* = {out['x']}")
    # print(out['cg'])
    print(f"iterations: {out['T']}")

    A, b = f1(sr, False)
    b = b.reshape(-1)

    Q = np.dot(A.T, A)
    new_b = np.dot(A.T, b)

    x0 = np.array([0 for _ in range(new_b.shape[0])])

    out = conjugate_gradient(Q, new_b, x0)
    print("#### PART4 ####")
    print(f"x* = {out['x']}")
    # print(out['cg'])
    print(f"iterations: {out['T']}")

    print(f"eigenvalues for ATA: {np.linalg.eigh(Q)[0]}")


if __name__ == "__main__":
    que1()