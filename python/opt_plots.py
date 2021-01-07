import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

def line(x, a, b):
    return a * x + b


def f_exp(x, a, b):
    return a * np.exp(-b * x)


def fit_func(x, *coeffs):
    y = np.polyval(coeffs, x)
    return y


def f_log(x, a, b):
    return a*np.log(b*x)


def plot_exp(data, opt_scatter, color, lbl):

    x = data[0, :]
    y = data[1, :]

    if opt_scatter:
        plt.scatter(x, y)

    popt, pcov = curve_fit(f_exp, x, y)
    a, b = popt
    x_fine = np.linspace(0., 21., 100)  # define values to plot the function for
    ax.plot(x_fine, f_exp(x_fine, popt[0], popt[1]), color+'-', label=lbl)


##################################################################################
data_nm = np.loadtxt('Test_data/nm_data.dat')
data_BFGS = np.loadtxt('Test_data/BFGS_data.dat')
data_shgo = np.loadtxt('Test_data/shgo_data.dat')
data_BOBYQA = np.loadtxt('Test_data/BOBYQA_data.dat')
data_BOBYQA_e8 = np.loadtxt('Test_data/BOBYQA_e8_data.dat')
data_COBYLA = np.loadtxt('Test_data/COBYLA_data.dat')


fig, ax = plt.subplots()

plot_exp(data_nm, False, 'b', 'nm')
plot_exp(data_shgo, False, 'r-', 'shgo')
plot_exp(data_BFGS, False, 'g', 'BFGS')
plot_exp(data_BOBYQA, False, 'k-', 'BOBYQA 1e-5')
plot_exp(data_BOBYQA_e8, False, 'c', 'BOBYQA 1e-8')
plot_exp(data_COBYLA, True, 'm', 'COBYLA')
leg = ax.legend();

plt.show()