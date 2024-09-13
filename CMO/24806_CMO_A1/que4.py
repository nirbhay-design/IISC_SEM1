from CMO_A1 import f1, f2, f3, f4 
import numpy as np 
import matplotlib.pyplot as plt 


def calculate_fx(x):
    return x * (x-1) * (x-3) * (x+2)

def plotx(mp):
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    iterations = [i for i in range(len(mp['f_at']))]
    axs[0,0].plot(iterations, mp['f_at'], label='f(at)')
    axs[0,0].set_ylabel('f(at)')
    axs[0,0].legend()


    axs[0,1].plot(iterations, mp['f_bt'], label='f(bt)')
    axs[0,1].set_ylabel('f(bt)')
    axs[0,1].legend()


    axs[1,0].plot(iterations, mp['bt_at'], label='bt-at')
    axs[1,0].set_ylabel('bt-at')
    axs[1,0].legend()


    axs[1,1].plot(iterations[:-1], mp['bt_at_bt1_at1'], label='bt-at/bt1-at1')
    axs[1,1].set_ylabel('bt-at / bt1-at1')
    axs[1,1].legend()
    plt.show()

def golden_section_search(a,b):
    phi = (1 + np.sqrt(5)) / 2
    rho = phi - 1
    x1 = rho * a + (1 - rho) * b
    x2 = (1 - rho) * a + rho * b
    interval = [a, x1, x2, b]
    interval_epsilon = 1e-4

    mp = {'f_at':[], 'f_bt':[], 'bt_at': [], 'fb_fa': [], 'bt_at_bt1_at1':[]}
    T = 0

    while abs(interval[-1] - interval[0]) > interval_epsilon:
        T += 1
        a,x1,x2,b = interval

        mp['f_at'].append(calculate_fx(a))
        mp['f_bt'].append(calculate_fx(b))
        if len(mp['bt_at']) >= 1:
            mp['bt_at_bt1_at1'].append((b-a) / mp['bt_at'][-1]) 
        mp['bt_at'].append(b - a)
        mp['fb_fa'].append(calculate_fx(b) - calculate_fx(a))
        

        fx1 = calculate_fx(x1)
        fx2 = calculate_fx(x2)
        if fx1 <= fx2:
            x3 = rho * a + (1-rho) * x2
            interval = [a, x3, x1, x2]
        elif fx2 <= fx1:
            x3 = (1-rho) * x1 + rho * b
            # x3 = rho * x1 + (1-rho) * b
            interval = [x1, x2, x3, b]

    plotx(mp)

    return interval, T

def golden_section_search_fibonacci(a,b):
    fib = [1, 1, 2]
    # T = 100
    T = 100
    for i in range(3, T + 2):
        fib.append(fib[i-1] + fib[i-2])

    rho = (fib[T] / fib[T + 1])
    # rho = 1 - (fib[T] / fib[T + 1])
    # x1 = (1 - rho) * a + rho * b
    # x2 = rho * a + (1 - rho) * b
    x1 = rho * a + (1 - rho) * b
    x2 = (1 - rho) * a + rho * b
    interval = [a, x1, x2, b]

    mp = {'f_at':[], 'f_bt':[], 'bt_at': [], 'fb_fa': [], 'bt_at_bt1_at1':[]}

    i = 1

    Iterations = 0

    while abs(interval[-1] - interval[0]) > (b - a) / fib[-1]:
        Iterations += 1
        a,x1,x2,b = interval

        mp['f_at'].append(calculate_fx(a))
        mp['f_bt'].append(calculate_fx(b))
        if len(mp['bt_at']) >= 1:
            mp['bt_at_bt1_at1'].append((b-a) / mp['bt_at'][-1]) 
        mp['bt_at'].append(b - a)
        mp['fb_fa'].append(calculate_fx(b) - calculate_fx(a))
        

        fx1 = calculate_fx(x1)
        fx2 = calculate_fx(x2)
        if fx1 <= fx2:
            # x3 = (1 - rho) * a + rho * x2
            x3 = rho * a + (1-rho) * x2
            interval = [a, x3, x1, x2]
        elif fx2 <= fx1:
            x3 = (1-rho) * x1 + rho * b
            # x3 = rho * x1 + (1 - rho) * b
            interval = [x1, x2, x3, b]

        # rho = (1 - fib[T - i]/ fib[T - i + 1])
        rho = fib[T - i]/ fib[T - i + 1]
        i += 1 

    plotx(mp)

    return interval, Iterations


if __name__ == "__main__":
    interval, T = golden_section_search(1,3)
    print(f"Golden section search: interval: {interval}, Iterations: {T}\n")
    interval, T = golden_section_search_fibonacci(1,3)
    print(f"Fibonacci search: interval: {interval}, Iterations: {T}")
