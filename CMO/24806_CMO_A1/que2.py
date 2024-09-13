from CMO_A1 import f1, f2, f3, f4 
import numpy as np 
import matplotlib.pyplot as plt 

SR = 24806

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

if __name__ == "__main__":
    x, fx, T = ConstantGradientDescent(1e-5, np.array([0.0 for _ in range(5)]))

    print(f"ConstGradientDescent: x = {x}, f(x) = {fx}, Iterations = {T}\n")

    x, fx = DiminishingGradientDescent(1e-3, np.array([0.0 for _ in range(5)]))
    
    print(f"DiminishingGradientDescent: x = {x}, f(x) = {fx}, Iterations = {int(1e4)}\n")


    x, fx, T = InExactLineSearch(0.2,0.8,0.5)

    print(f"InexactLineSearch: x = {x}, f(x) = {fx}, Iterations = {T}\n")


    x, fx, T = ExactLineSearch()

    print(f"ExactLineSearch: x = {x}, f(x) = {fx}, Iterations = {T}\n")
