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
        plt.scatter(x, y, c=color, label=lbl)

    popt, pcov = curve_fit(f_exp, x, y)
    a, b = popt
    x_fine = np.linspace(0., 21., 100)  # define values to plot the function for
    #ax.plot(x_fine, f_exp(x_fine, popt[0], popt[1]), color+'-', label=lbl)


def av(lst):
    return sum(lst) / len(lst)


def plt_scatter(data, color, lbl):

    x = data[0, :]
    y = data[1, :]
    sum_arr = [0]*int(max(x)+1)
    count = [0]*int(max(x)+1)

    size_arr = range(1, int(max(x))-2)


    for val in range(len(x)):
        sum_arr[int(x[val])] += y[val]  # sum_arr ranges from 0 to max(x)
        count[int(x[val])] += 1
    # sum[3] contains values for x=3
    sum_arr = sum_arr[3:]
    count_arr = count[3:]
    av_arr = [0]*len(count_arr)

    for k in range(len(count_arr)):
        av_arr[k] = sum_arr[k]/count_arr[k]
    y_new = av_arr
    x_new = range(3, int(max(x))+1)
    ax_acc.scatter(x_new, y_new, c=color, label=lbl)
    plt.xticks(np.arange(min(x_new) - 1, max(x_new) + 1, 1.0))




##################################################################################
#data_nm = np.loadtxt('Test_data/nm_data.dat')
#data_BFGS = np.loadtxt('Test_data/BFGS_data.dat')
#data_shgo = np.loadtxt('Test_data/shgo_data.dat')
#data_BOBYQA = np.loadtxt('Test_data/BOBYQA_e5_data.dat')
#data_BOBYQA_e8 = np.loadtxt('Test_data/BOBYQA_e8_data.dat')
#data_COBYLA = np.loadtxt('Test_data/COBYLA_data.dat')
#data_shgo_e8 = np.loadtxt('Test_data/shgo_e-8_data.dat')

#fig, ax = plt.subplots()

# plot_exp(data_nm, False, 'b', 'nm')
# plot_exp(data_shgo, False, 'r', 'shgo 1e-8')
# plot_exp(data_BFGS, False, 'g', 'BFGS')
# plot_exp(data_BOBYQA, False, 'k-', 'BOBYQA 1e-5')
# plot_exp(data_BOBYQA_e8, False, 'c', 'BOBYQA 1e-8')
# plot_exp(data_COBYLA, False, 'm', 'COBYLA')
# leg = ax.legend()
# plt.xlabel("Problem Size [vertices]")
# plt.ylabel("Run-time [s]")
# plt.show()

# acc_data_nm = np.loadtxt('Test_data/Accuracy/nm_rel_acc_data.dat')
# acc_data_bfgs = np.loadtxt('Test_data/Accuracy/bfgs_rel_acc_data.dat')
# acc_data_shgo = np.loadtxt('Test_data/Accuracy/shgo_rel_acc_data.dat')
# acc_data_bobyqa = np.loadtxt('Test_data/Accuracy/bobyqa_rel_acc_data.dat')
# acc_data_cobyla = np.loadtxt('Test_data/Accuracy/cobyla_rel_acc_data.dat')
# acc_data_r_cobyla = np.loadtxt('Test_data/Accuracy/r_cobyla_rel_acc_data.dat')
#
# fig_acc, ax_acc = plt.subplots()
#
# plt_scatter(acc_data_nm, 'b', 'nm')
# plt_scatter(acc_data_bfgs, 'g', 'BFGS')
# plt_scatter(acc_data_shgo, 'r', 'SHGO')
# plt_scatter(acc_data_bobyqa, 'c', 'BOBYQA')
# plt_scatter(acc_data_cobyla, 'm', 'COBYLA')
# plt_scatter(acc_data_r_cobyla, 'k', 'R COBYLA')
# ax_acc.legend()
# plt.xlabel("Problem Size [vertices]")
# plt.ylabel("Relative Accuracy")
#
# plt.show()


data_MCP = np.loadtxt('Test_data/shgo_data.dat')
data_DSP = np.loadtxt('Test_data/dsptiming_data.dat')
data_TSP = np.loadtxt('Test_data/tsptiming_data.dat')

fig, ax = plt.subplots()

plot_exp(data_MCP, True, 'k', 'MCP')
plot_exp(data_DSP, True, 'b', 'DSP')
plot_exp(data_TSP, True, 'r', 'TSP')

#ax_app.scatter(data_DSP, c='b', label='DSP')
#ax_app.scatter(data_TSP, c='r', label='TSP')


ax.legend()
plt.xlabel("Problem Size [vertices]")
plt.ylabel("run-time [s]")

plt.show()
