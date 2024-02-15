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
    
    fig = plt.figure(figsize=(14,8))
    gs  = fig.add_gridspec(100,100)
    ax1 = fig.add_subplot(gs[0:42,   0:26])
    ax2 = fig.add_subplot(gs[0:42,  36:62])
    ax3 = fig.add_subplot(gs[0:42,  72:100])
    ax4 = fig.add_subplot(gs[58:100, 0:26])
    ax5 = fig.add_subplot(gs[58:100,36:62])
    ax6 = fig.add_subplot(gs[58:100,72:100])

    return fig, ax1, ax2, ax3, ax4, ax5, ax6


def plot_show(time):
    plt.pause(time)


def plot_close():
    plt.close()


def plot_save(path, name):
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


def compare_error(dspin_2305, dspin_unet, ax):
    ax.cla()
    ax.plot( dspin_2305[0], dspin_2305[1], color='darkgreen',  alpha=0.9, label='2305' )
    ax.plot( dspin_unet[0], dspin_unet[1], color='darkorange', alpha=0.9, label='unet' )
    ax.set_ylim(1.0e-5, 1)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"Max $\Delta M / M_s$")
    ax.set_yscale("log", base=10)
    ax.legend()
    ax.set_title("Spin change")


def compare_wind(wind_2305, wind_unet, ax):
    ax.cla()
    ax.plot( wind_2305[0], wind_2305[1], color='green',  alpha=0.7, label='2305' )
    ax.plot( wind_unet[0], wind_unet[1], color='orange', alpha=0.7, label='unet' )
    ax.set_ylim(0,10)
    ax.grid(True, axis='y',lw=1.0, ls='-.')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Absolute winding number")
    ax.legend()
    ax.set_title("Vortex cores")
