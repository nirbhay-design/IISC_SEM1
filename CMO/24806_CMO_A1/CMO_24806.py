from CMO_A1 import f1, f2, f3, f4 
import numpy as np 
import matplotlib.pyplot as plt 

SR = 24806

###################################################que-1######################################################

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

###################################################que-2######################################################

class MeterUpdate:
    def __init__(self):
        self.mp = {"gradf": [], "fx": [], "xk": []}
    
    def update(self, gradf, fx, xk):
        self.mp['gradf'].append(gradf)
        self.mp['fx'].append(fx)
        self.mp['xk'].append(xk)

    def calculate(self):
        fxT = self.mp['fx'][-1]
        xT = self.mp['xk'][-1]
        self.mp['fx_fxT'] = [i - fxT for i in self.mp['fx'][:-1]]
        self.mp['x_xT_norm'] = [np.linalg.norm(i - xT) ** 2 for i in self.mp['xk'][:-1]]

        try:
            f_diff = self.mp['fx_fxT']
            self.mp['f_ratio'] = [f_diff[i] / f_diff[i-1] for i in range(1, len(f_diff))]

            x_xT = self.mp['x_xT_norm']
            self.mp['norm_ratio'] = [x_xT[i] / x_xT[i-1] for i in range(1, len(x_xT))]
        except:
            self.mp['f_ratio'] = []
            self.mp['norm_ratio'] = []

    def plotx(self):
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        axs[0,0].plot([i for i in range(len(self.mp['gradf']))], self.mp['gradf'], label='gradf')
        axs[0,0].set_ylabel('gradf')
        axs[0,0].set_xlabel('iterations')
        axs[0,0].legend()

        axs[0,1].plot([i for i in range(len(self.mp['fx_fxT']))], self.mp['fx_fxT'], label='fx-fxT')
        axs[0,1].set_ylabel('fx-fxT')
        axs[0,1].set_xlabel('iterations')
        axs[0,1].legend()

        if self.mp['f_ratio']:

            axs[1,0].plot([i for i in range(len(self.mp['f_ratio']))], self.mp['f_ratio'], label='fxk-fxT/fx(k-1)-fxT')
            axs[1,0].set_ylabel('f_ratio')
            axs[1,0].set_xlabel('iterations')

            axs[1,0].legend()
        else:
            axs[1,0].set_axis_off()

        axs[1,1].plot([i for i in range(len(self.mp['x_xT_norm']))], self.mp['x_xT_norm'], label='||x-xT||**2')
        axs[1,1].set_ylabel('||x-xT||**2')
        axs[1,1].set_xlabel('iterations')
        axs[1,1].legend()

        if self.mp['norm_ratio']:

            axs[0,2].plot([i for i in range(len(self.mp['norm_ratio']))], self.mp['norm_ratio'], label='||x-xT||**2/||x(k-1)-xT||**2')
            axs[0,2].set_ylabel('norm_ratio')
            axs[0,2].set_xlabel('iterations')
            axs[0,2].legend()
        else:
            axs[0,2].set_axis_off()

        axs[1,2].set_axis_off()
        fig.tight_layout()
        plt.show()

def ConstantGradientDescent(alpha, initialx):
    x = initialx.copy()
    epsilon = 1e-5
    optimal_fx = None
    meter = MeterUpdate()
    T = 0
    while True:
        T += 1
        fx, gradfx = f4(SR, x)
        grad_norm = np.linalg.norm(gradfx)
        meter.update(grad_norm, fx, x)
        if  grad_norm <= epsilon:
            optimal_fx = fx
            break 
        x = (x - alpha * gradfx).copy()
    meter.calculate()
    meter.plotx()
    return x, optimal_fx, T

def DiminishingGradientDescent(InitialAlpha, initialx):
    x = initialx.copy()
    alpha = InitialAlpha
    meter = MeterUpdate()
    for i in range(1, int(1e4) + 1):
        fx, gradfx = f4(SR, x)
        meter.update(np.linalg.norm(gradfx), fx, x)
        x = (x - alpha * gradfx).copy()
        alpha = InitialAlpha / (i + 1)
    optimal_fx, grad_opt_fx = f4(SR, x)
    meter.calculate()
    meter.plotx()
    return x, optimal_fx

def ArmijoWolfeCondition(xk, fxk, gradfxk, alphak, c1=0.6, c2=0.4):
    pk = -gradfxk
    xalphap = xk + alphak * pk
    fxap, gradfxap = f4(SR, xalphap)
    pktfx = np.dot(pk, gradfxk)
    fcl = fxap
    fcr = fxk + c1 * alphak * pktfx 
    scl = -np.dot(pk, gradfxap)
    scr = -c2 * pktfx
    return (fcl <= fcr) and (scl <= scr)

def InExactLineSearch(c1, c2, gamma):
    x = np.array([0.0 for _ in range(5)])
    epsilon = 1e-4
    optimal_fx = None
    meter = MeterUpdate()
    T = 0
    while True:
        T += 1
        fx, gradfx = f4(SR, x)
        grad_norm = np.linalg.norm(gradfx)
        meter.update(grad_norm, fx, x)
        if grad_norm <= epsilon:
            optimal_fx = fx
            break 
        alpha = 1
        while not ArmijoWolfeCondition(x, fx, gradfx, alpha, c1, c2):
            alpha = alpha * gamma
        x = (x - alpha * gradfx).copy()
    meter.calculate()
    meter.plotx()
    return x, optimal_fx, T

def findb():
    a = np.array([0.0 for _ in range(5)])
    fx, gradfx = f4(SR, a)
    return gradfx 

def ExactLineSearch():
    x = np.array([0.0 for _ in range(5)])
    epsilon = 1e-4
    optimal_fx = None
    b = findb()
    meter = MeterUpdate()
    T = 0
    while True:
        T += 1
        fx, gradfx = f4(SR, x)
        grad_norm = np.linalg.norm(gradfx)
        meter.update(grad_norm, fx, x)
        if grad_norm <= epsilon:
            optimal_fx = fx
            break 
        pk = -gradfx
        alpha = -np.dot(gradfx, pk) / (2 * (f4(SR, pk)[0] - np.dot(b,pk)))
        x = (x - alpha * gradfx).copy()
    meter.calculate()
    meter.plotx()
    return x, optimal_fx, T

###################################################que-3######################################################

def f(x):
    x1,x2 = x
    expx1x2 = np.exp(x1 * x2)
    gradf = np.array([x2 * expx1x2, x1 * expx1x2])
    return expx1x2, gradf

# que 3

def draw_contour():
    x = np.arange(-1.0, 1.0, 0.01) 
    y = np.arange(-1.0, 1.0, 0.01) 
    
    [X, Y] = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(1, 1) 
    
    Z = np.exp(X*Y)
    
    ax.contourf(X, Y, Z) 
    
    ax.set_title('Contour of e^{xy}') 
    ax.set_xlabel('x') 
    ax.set_ylabel('y') 
    
    plt.show() 

def draw_contour_points(iterx, fx):
    x = np.arange(-1.0, 1.0, 0.01) 
    y = np.arange(-1.0, 1.0, 0.01) 
    
    [X, Y] = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    
    Z = np.exp(X*Y)
    
    x1, x2 = iterx
    
    ax[0].contourf(X, Y, Z) 
    ax[0].plot(x1, x2, label = 'trajectry', c = 'r')
    
    ax[0].set_title('Contour of e^{xy}') 
    ax[0].set_xlabel('x') 
    ax[0].set_ylabel('y') 
    ax[0].legend()

    ax[1].plot([i for i in range(len(fx))], fx, label='fx')
    ax[1].set_xlabel('iterations')
    ax[1].set_ylabel('fx')
    ax[1].legend()

    fig.tight_layout()
    
    plt.show() 


def ConstantGradientDescentq3(alpha, initialx):
    x = initialx.copy()
    epsilon = 1e-3
    optimal_fx = None

    iter_x = [[],[]]
    fx_vals = []

    while True:
        fx, gradfx = f(x)
        fx_vals.append(fx)
        if np.linalg.norm(gradfx) <= epsilon:
            optimal_fx = fx 
            break 
        x = (x - alpha * gradfx).copy()

        iter_x[0].append(x[0])
        iter_x[1].append(x[1])

    return x, optimal_fx, iter_x, fx_vals

def DiminishingGradientDescentq3(InitialAlpha, initialx):
    x = initialx.copy()
    epsilon = 1e-3
    optimal_fx = None

    iter_x = [[],[]]
    fx_vals = []

    alpha = InitialAlpha

    for i in range(1,10001):
        fx, gradfx = f(x)
        fx_vals.append(fx)
        if np.linalg.norm(gradfx) <= epsilon:
            optimal_fx = fx 
            break 
        x = (x - alpha * gradfx).copy()
        alpha = InitialAlpha / (i + 1)

        iter_x[0].append(x[0])
        iter_x[1].append(x[1])

    return x, optimal_fx, iter_x, fx_vals

def get_expected_f_value(x, f, gradfx, alpha, var):
    avg = 0
    for i in range(10):
        new_x = x - alpha * (gradfx + get_noise(var, 2))
        avg += f(new_x)[0] / 10
    return avg

def GradientDescentConstAlphaConstVar(alpha, var, initialx):
    x = initialx.copy()
    epsilon = 1e-3
    optimal_fx = None

    iter_x = [[],[]]
    fx_vals = []

    for i in range(1,int(1e5) + 1):
        fx, gradfx = f(x)
        if np.linalg.norm(gradfx) <= epsilon:
            optimal_fx = fx 
            break 

        fx_vals.append(get_expected_f_value(x, f, gradfx, alpha, var))
        x = (x - alpha * (gradfx + get_noise(var, 2))).copy()

        iter_x[0].append(x[0])
        iter_x[1].append(x[1])

    return x, optimal_fx, iter_x, fx_vals

def GradientDescentConstAlphaDecVar(alpha, initvar, initialx):
    x = initialx.copy()
    epsilon = 1e-3
    optimal_fx = None

    iter_x = [[],[]]
    fx_vals = []

    var = initvar

    for i in range(1,int(1e5) + 1):
        fx, gradfx = f(x)
        if np.linalg.norm(gradfx) <= epsilon:
            optimal_fx = fx 
            break 
        fx_vals.append(get_expected_f_value(x, f, gradfx, alpha, var))
        x = (x - alpha * (gradfx + get_noise(var, 2))).copy()


        iter_x[0].append(x[0])
        iter_x[1].append(x[1])

        var = initvar / (i + 1)

    return x, optimal_fx, iter_x, fx_vals

def GradientDescentDecAlphaConstVar(initalpha, var, initialx):
    x = initialx.copy()
    epsilon = 1e-3
    optimal_fx = None

    iter_x = [[],[]]
    fx_vals = []

    alpha = initalpha

    for i in range(1,int(1e5) + 1):
        fx, gradfx = f(x)
        if np.linalg.norm(gradfx) <= epsilon:
            optimal_fx = fx 
            break 
        fx_vals.append(get_expected_f_value(x, f, gradfx, alpha, var))
        x = (x - alpha * (gradfx + get_noise(var, 2))).copy()

        iter_x[0].append(x[0])
        iter_x[1].append(x[1])

        alpha = initalpha / (i + 1)

    return x, optimal_fx, iter_x, fx_vals

def GradientDescentDecAlphaDecVar(initalpha, initvar, initialx):
    x = initialx.copy()
    epsilon = 1e-3
    optimal_fx = None

    iter_x = [[],[]]
    fx_vals = []

    alpha = initalpha
    var = initvar

    for i in range(1,int(1e5) + 1):
        fx, gradfx = f(x)
        if np.linalg.norm(gradfx) <= epsilon:
            optimal_fx = fx 
            break 
        fx_vals.append(get_expected_f_value(x, f, gradfx, alpha, var))
        x = (x - alpha * (gradfx + get_noise(var, 2))).copy()

        iter_x[0].append(x[0])
        iter_x[1].append(x[1])

        alpha = initalpha / (i + 1)
        var = initvar / (i + 1)

    return x, optimal_fx, iter_x, fx_vals

def get_noise(var, dim):
    mu = np.zeros(dim)
    sigma = var * np.eye(dim)

    return np.random.multivariate_normal(mu, sigma, 1)[0]

###################################################que-4######################################################


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
    T = 100
    for i in range(3, T + 2):
        fib.append(fib[i-1] + fib[i-2])

    rho = 1 - (fib[T] / fib[T + 1])
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
            x3 = rho * a + (1-rho) * x2
            interval = [a, x3, x1, x2]
        elif fx2 <= fx1:
            x3 = (1-rho) * x1 + rho * b
            interval = [x1, x2, x3, b]

        rho = (1 - fib[T - i]/ fib[T - i + 1])
        i += 1 

    plotx(mp)

    return interval, Iterations


def run_que1():
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

def run_que2():
    x, fx, T = ConstantGradientDescent(1e-5, np.array([0.0 for _ in range(5)]))

    print(f"ConstGradientDescent: x = {x}, f(x) = {fx}, Iterations = {T}\n")

    x, fx = DiminishingGradientDescent(1e-3, np.array([0.0 for _ in range(5)]))
    
    print(f"DiminishingGradientDescent: x = {x}, f(x) = {fx}, Iterations = {int(1e4)}\n")


    x, fx, T = InExactLineSearch(0.2,0.8,0.5)

    print(f"InexactLineSearch: x = {x}, f(x) = {fx}, Iterations = {T}\n")


    x, fx, T = ExactLineSearch()

    print(f"ExactLineSearch: x = {x}, f(x) = {fx}, Iterations = {T}\n")

def run_que3():
    draw_contour()

    x, optimalf, iterx, fx_vals = ConstantGradientDescentq3(1e-5, np.array([0.99,0.99]))
    draw_contour_points(iterx, fx_vals)
    x, optimalf, iterx, fx_vals = DiminishingGradientDescentq3(1e-3, np.array([0.5,0.5]))
    draw_contour_points(iterx, fx_vals)



    

    # x, optimalf, iterx, fx_vals = GradientDescentConstAlphaConstVar(1e-5, 1e-5, np.array([0.9,0.9]))
    # draw_contour_points(iterx, fx_vals)
    # x, optimalf, iterx, fx_vals = GradientDescentConstAlphaDecVar(1e-5, 1e-5, np.array([0.9,0.9]))
    # draw_contour_points(iterx, fx_vals)
    # x, optimalf, iterx, fx_vals = GradientDescentDecAlphaDecVar(1e-1, 1e-1, np.array([0.9,0.9]))
    # draw_contour_points(iterx, fx_vals)
    # x, optimalf, iterx, fx_vals = GradientDescentDecAlphaConstVar(1e-1, 1e-1, np.array([0.9,0.9]))
    # draw_contour_points(iterx, fx_vals)

def run_que4():
    interval, T = golden_section_search(1,3)
    print(f"Golden section search: interval: {interval}, Iterations: {T}\n")
    interval, T = golden_section_search_fibonacci(1,3)
    print(f"Fibonacci search: interval: {interval}, Iterations: {T}")


if __name__ == "__main__":
    print("possible values of input: q1, q2, q3, q4")

    que = input('which question output do you want: ')
    fun_map = {
        "q1": run_que1,
        "q2": run_que2,
        "q3": run_que3,
        "q4": run_que4,
    }

    fun_map[que]()