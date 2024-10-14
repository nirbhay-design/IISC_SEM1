from oracles import f1, f2, f3
import numpy as np 
import matplotlib.pyplot as plt 
import warnings; warnings.filterwarnings("ignore")

sr = 24806

def gradf(A, x, b):
    return np.dot(A,x) - b

def plot_values(output, label_name='alpha', ylim = None, newton = False):
    if newton:
        for out in output:
            fval = out['f']
            alpha = out.get('a', None)
            K = out['K']
            plt.plot([i for i in range(K)], fval[:K], marker = 'o', label=f'{label_name}: {alpha}' if alpha is not None else "")
            plt.plot([i for i in range(K, len(fval))], fval[K:], marker='x', label=f'{label_name}: {alpha}' if alpha is not None else "")
    else:
        for out in output:
            fval = out['f']
            alpha = out.get('a', None)
            plt.plot([i for i in range(len(fval))], fval, label=f'{label_name}: {alpha}' if alpha is not None else "")
    plt.xlabel("Iterations")
    plt.ylabel('f(x)')
    plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.show()

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

def ConstantGradientDescent(fun, alpha, initialx):
    x = initialx
    fxvals = []
    for i in range(100):
        fx = fun(x, sr, 0)
        gradfx = fun(x, sr, 1)
        x = x - alpha * gradfx
        fxvals.append(fx)
    return {"f": fxvals, "a": alpha, 'x': x}

def NewtonMethod(fun, initialx, scale=False):
    x = initialx
    fxvals = []
    for i in range(100):
        fx = fun(x, sr, 0)
        u = fun(x, sr, 2)
        if scale:
            u = u / 1e6
        x = x - u
        fxvals.append(fx)
    return {"f": fxvals, 'x': x}

def GDNewton(fun, alpha, initialx, K):
    x = initialx
    fxvals = []
    for i in range(K):
        fx = fun(x, sr, 0)
        gradfx = fun(x, sr, 1)
        x = x - alpha * gradfx
        fxvals.append(fx)
    for i in range(K, 100):
        fx = fun(x, sr, 0)
        u = fun(x, sr, 2)
        x = x - u
        fxvals.append(fx)
    return {"f": fxvals, "a": alpha, "K":K, 'x': x}

def quasiNewtonELS(fun, initialx):
    x = initialx
    G = np.eye(x.shape[0])
    g = fun(x, sr, 1)
    fvals = []
    b = fun(np.zeros(x.shape[0]), sr, 1)
    T = 0
    while True:
        T += 1
        if np.linalg.norm(g) < 1e-3:
            break;
        fvals.append(fun(x, sr, 0))
        u = -np.dot(G, g)
        alpha = -np.dot(g, u) / (2 * (fun(u, sr, 0) - np.dot(b, u)))
        delta = alpha * u
        x = x + delta
        gnew = fun(x, sr, 1)
        gamma = gnew - g
        term = delta - np.dot(G, gamma) 
        G = G + np.dot(term.reshape(-1,1), term.reshape(-1,1).T) / np.dot(term, gamma)
        g = gnew
    return {'f': fvals, 'x': x, 'T':T}

def quasiNewtonIdELS(fun, initialx):
    x = initialx
    G = np.eye(x.shape[0])
    g = fun(x, sr, 1)
    fvals = []
    b = fun(np.zeros(x.shape[0]), sr, 1)
    for i in range(100): 
        # if np.linalg.norm(g) < 1e-3:
        #     break;
        fvals.append(fun(x, sr, 0))
        u = -np.dot(G, g)
        alpha = -np.dot(g, u) / (2 * (fun(-u, sr, 0) - np.dot(b, -u)))
        delta = alpha * u
        x = x + delta
        gnew = fun(x, sr, 1)
        gamma = gnew - g
        G = (np.dot(delta, delta) / np.dot(delta, gamma)) * np.eye(x.shape[0])
        g = gnew
    return {'f': fvals, 'x': x}

def que1():
    A, b = f1(sr, True)
    b = b.reshape(-1)
    x0 = np.array([0 for _ in range(b.shape[0])])
    out = conjugate_gradient(A, b, x0)
    print("#### Que1.2 ####")
    print(f"x* = {out['x']}")
    # print(out['cg'])
    print(f"iterations: {out['T']}")

    A, b = f1(sr, False)
    b = b.reshape(-1)

    Q = np.dot(A.T, A)
    new_b = np.dot(A.T, b)

    x0 = np.array([0 for _ in range(new_b.shape[0])])

    out = conjugate_gradient(Q, new_b, x0)
    print("#### Que1.4 ####")
    print(f"x* = {out['x']}")
    # print(out['cg'])
    print(f"iterations: {out['T']}")

    print(f"eigenvalues for ATA: {np.linalg.eigh(Q)[0]}")


def que2():
    x = np.array([0.0 for _ in range(5)])
    alphas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    outputs = []
    for alpha in alphas:
        outputs.append(ConstantGradientDescent(f2, alpha, x))
    plot_values(outputs)

    print("#### Que-2.1 ####")
    for out in outputs:
        print(f"alpha: {out['a']}, x* = {out['x']}")

    out = NewtonMethod(f2, x)
    plot_values([out])
    print("#### Que-2.2 ####")
    print(f"Newton x* = {out['x']}")

    newton_outputs = []
    initialxs = [np.array([2,4,8,0,6]), np.array([2,4,6,2,2]), np.array([2,4,0,1,6]), np.array([2,4,0,1,4]), np.array([1,2,3,4,5])]
    for x in initialxs:
        newton_out = NewtonMethod(f2, x)
        newton_out['a'] = x
        newton_outputs.append(newton_out)
    plot_values(newton_outputs, "x0")
    print("#### Que-2.3 ####")
    for idx, nout in enumerate(newton_outputs):
        print(f"x0: {initialxs[idx]}, x*: {nout['x']}")

def find_min_index(lis):
    idx = 0
    mn = lis[idx]
    for i in range(1, len(lis)):
        if mn > lis[i]:
            idx = i 
            mn = lis[idx]
    return mn, idx

def que3():
    x0 = np.array([1,1,1,1,1],dtype=np.float32)

    out1 = ConstantGradientDescent(f3, 0.1, x0)
    plot_values([out1])
    print("#### Que-3.1 ####")
    print(f"min f(x): {min(out1['f']):.3f}")


    print("#### Que-3.2 ####")
    alphas = [0.1,0.2,0.3,0.4,0.5]
    outs = []
    for alpha in alphas:
        out = ConstantGradientDescent(f3, alpha, x0)
        outs.append(out)
    plot_values(outs)

    out2 = []
    for a in [0.02, 0.01, 0.009]:
        cout = ConstantGradientDescent(f3, a, x0)
        out2.append(cout)
        print(f"min f(x) (alpha = {a}): {min(cout['f']):.3f}")
    plot_values(out2)

    # out2 = ConstantGradientDescent(f3, 0.01, x0)
    # plot_values([out2])
    # print(f"min f(x): {min(out2['f']):.3f}")


    # out = NewtonMethod(f3, x0, scale=True)
    out = NewtonMethod(f3, x0)

    print("#### Que-3.2 ####")
    print(f"f(x) values for Newton")
    print(out['f'][:10])
    # print(f"f(x) values for Newton (alpha = 1e-6)")
    # print([round(i, 3) for i in out['f'][:10]])

    alphas = [0.01, 0.02, 0.1, 0.2]
    Ks = [60, 80, 85, 90, 100]

    all_k_alpha_fx = []

    for K in Ks:
        out_alpha = []
        for alpha in alphas:
            cgdnewton = GDNewton(f3, alpha, x0, K)
            cgdnewton['mnf'] = min(cgdnewton['f'])
            out_alpha.append(cgdnewton)
        plot_values(out_alpha, f"K: {K} alpha", ylim=[0,30], newton=True)
        all_k_alpha_fx.extend(out_alpha)

    # Ks = [10, 20, 30, 40, 50]
    # alphas = [0.01, 0.02, 0.009, 0.008]
    # all_k_alpha_fx = []

    # for K in Ks:
    #     out_alpha = []
    #     for alpha in alphas:
    #         cgdnewton = GDNewtonInterleve(f3, alpha, x0, K)
    #         cgdnewton['mnf'] = min(cgdnewton['f'])
    #         out_alpha.append(cgdnewton)
    #     plot_values(out_alpha, f"K: {K} alpha", ylim=[0,30])
    #     all_k_alpha_fx.extend(out_alpha)

    min_f = all_k_alpha_fx[0]['mnf']
    min_out = all_k_alpha_fx[0]
    for out_f in all_k_alpha_fx[1:]:
        if min_f > out_f['mnf']:
            min_f = out_f['mnf']
            min_out = out_f 
    print("#### Que-3.4 ####")
    print(f"min f(x): {min_out['mnf']:.3f}, K: {min_out['K']}, alpha:: {min_out['a']}")    

def que4():
    alphas = [1e-1, 2e-1, 1e-2, 3e-2, 1e-3]
    outs = []
    print("#### Gradient Descent ####")
    for alpha in alphas:
        out = ConstantGradientDescent(f2, alpha, np.zeros(5))
        print(f"alpha: {out['a']} x*: {out['x']}")
        outs.append(out)
    plot_values(outs)

    print("#### Quasi Newton Rank 1 ####")
    out = quasiNewtonELS(f2, np.zeros(5))
    print(f"Quasi Newton (alpha = Exact Line Search) x*: {out['x']}")
    plot_values([out])


    print("#### Quasi Newton scaler multiple of Identity ####")
    out = quasiNewtonIdELS(f2, np.zeros(5))
    print(f"Quasi Newton (alpha = Exact Line Search) x*: {[round(i, 3) for i in out['x']]}")
    plot_values([out])


if __name__ == "__main__":
    run_map = {
        "q1": que1,
        "q2": que2,
        "q3": que3,
        "q4": que4
    }

    que = int(input("which question output you want: [1/2/3/4]: "))
    run_map[f'q{que}']()

"""
def quasiNewtonConstStep(fun, alpha, initialx):
    x = initialx
    G = np.eye(x.shape[0])
    g = fun(x, sr, 1)
    fvals = []
    for i in range(100):
        fvals.append(fun(x, sr, 0))
        u = -np.dot(G, g)
        delta = alpha * u
        x = x + delta
        gnew = fun(x, sr, 1)
        gamma = gnew - g
        term = delta - np.dot(G, gamma) 
        G = G + np.dot(term.reshape(-1,1), term.reshape(-1,1).T) / np.dot(term, gamma)
        g = gnew
    return {'f': fvals, 'x': x, 'a': alpha}

def quasiNewtonId(fun, alpha, initialx):
    x = initialx
    G = np.eye(x.shape[0])
    g = fun(x, sr, 1)
    fvals = []
    for i in range(100):
        fvals.append(fun(x, sr, 0))
        u = -np.dot(G, g)
        delta = alpha * u
        x = x + delta
        gnew = fun(x, sr, 1)
        gamma = gnew - g
        G = (np.dot(delta, delta) / np.dot(delta, gamma)) * np.eye(x.shape[0])
        g = gnew
    return {'f': fvals, 'x': x, 'a': alpha}

    # outs = []
    # print("#### Quasi Newton Rank 1 ####")
    # for alpha in alphas:
    #     out = quasiNewtonConstStep(f2, alpha, np.zeros(5))
    #     outs.append(out)
    #     print(f"alpha: {out['a']} x*: {out['x']}")
    # plot_values(outs)

    # outs = []
    # print("#### Quasi Newton scaler multiple of Identity ####")
    # for alpha in alphas:
    #     out = quasiNewtonId(f2, alpha, np.zeros(5))
    #     print(f"alpha: {out['a']} x*: {out['x']}")
    #     outs.append(out)
    # plot_values(outs)
"""
