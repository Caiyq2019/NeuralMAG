import matplotlib.pyplot as plt
import numpy as np

# Font size
def plot_prepare():
    parameters = {'axes.labelsize' : 14,
                  'axes.titlesize' : 14,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'legend.fontsize': 11,
                  'figure.dpi'     : 150}
    plt.rcParams.update(parameters)
    
    fig, axs = plt.subplots(figsize=(10,4), ncols=3, nrows=1)

    return fig, axs.flat[0], axs.flat[1], axs.flat[2]


def plot_show(time):
    plt.pause(time)


def plot_close():
    plt.close()


def plot_save(path, name):
    plt.tight_layout()
    plt.savefig(path + name)


def plot_spin(spin, ax, title):
    ax.cla()
    spin_plt = (np.array(spin) + 1)/2  # spin range (-1,1)
    ax.imshow( spin_plt.transpose(1,0,2), origin='lower' )
    ax.set_aspect( "equal", adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)


def plot_wind(wind, ax, title):
    ax.cla()
    wind_plt = np.array(wind)
    ax.imshow( wind_plt.T, origin='lower',
               vmin=-0.25, vmax=0.25, alpha=1.0 )
    ax.set_aspect( "equal", adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)


def plot_wind_list(wind_2305, ax):
    ax.cla()
    ax.plot( wind_2305[0], wind_2305[1], color='green',  alpha=0.7, label='cores' )
    ax.set_ylim(0,10)
    ax.grid(True, axis='y',lw=1.0, ls='-.')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Absolute winding number")
    ax.legend()
    ax.set_title("Vortex cores")
