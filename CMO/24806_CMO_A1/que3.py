import numpy as np 
import matplotlib.pyplot as plt 

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

if __name__ == "__main__":
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