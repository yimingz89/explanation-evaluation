import numpy as np
import matplotlib.pyplot as plt

NUM_DATA = 10
NUM_EPOCHS = 60
PLOT_FREQ = 4
NUM_CONFIGS=7
COLORS = ['red' ,'orange', 'green', 'cyan', 'blue', 'magenta', 'pink']

def plot_results():
    x = np.load("./results/accuracy-curve-0.npy")
    data = np.zeros((NUM_DATA, x.shape[0], x.shape[1]))
    for i in range(NUM_DATA):
        data[i] = np.load("./results/accuracy-curve-" + str(i) + ".npy")
    data = np.mean(data, axis=0)
    
    fig, ax = plt.subplots()
    x = np.arange(0,NUM_EPOCHS+PLOT_FREQ,PLOT_FREQ)
    for i in range(NUM_CONFIGS):
        if i == 0:
            lam = '0'
        else:
            lam = '1e' + str(-2*i)
        ax.plot(x, data[i], color=COLORS[i % len(COLORS)], label=lam)
    leg = ax.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
    plt.savefig('./results/accuracy-curves.pdf', bbox_inches='tight')

if __name__ == "__main__":
    plot_results()
