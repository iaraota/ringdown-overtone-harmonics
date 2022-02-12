import numpy as np
from numpy import arctan, pi
import matplotlib.pyplot as plt
import matplotlib as mpl

class Plots:

    def __init__(
        self,
        ):
        pass

    def plot_d_theta(
        self,
        label_dominant:str,
        label_mode:str,
        w_fit:float,
        w_simu:float,
        ):
        # import theta dot
        d_arctan = np.genfromtxt(f'data/d_theta/peak_{label_dominant}_{label_mode}.dat', delimiter='\t')
        t_final = np.genfromtxt(f'data/times_fundamental.dat', delimiter='\t')[1]
        d_arctan = np.array(d_arctan)

        # plot
        plt.close('all')

        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['figure.figsize'] = [12, 8]  # plot image size

        font_size = 40
        
        f, sp = plt.subplots(1)
        sp.set_facecolor('#FDFDFD')
        sp.set_xlim(0, (int(t_final/10)+1)*10)
        sp.set_ylim(d_arctan[0][1],max(max(d_arctan[:int(((int(t_final/10)+1)*10)/0.1),1]), w_simu, w_fit)*1.01)
        sp.autoscale_view()
        sp.set_xlabel(r'$t-t_\mathrm{peak}[M]$', fontsize=font_size)
        sp.set_ylabel(r'$\dot{\theta}_{%s}[1/M]$'%('{'+label_mode+'}'), fontsize=font_size, labelpad=15)
        sp.tick_params(axis='both', which='major', labelsize=font_size)
        sp.axhline(y=w_fit, color='limegreen', linewidth=2.5, ls = '--', label=r'$\omega^r_%s$ $\mathrm{(fit)}$'%('{'+label_mode+'n0}'))
        sp.axhline(y=w_simu, color='darkgreen', linewidth=2.5, label=r'$\omega^r_%s$ $\mathrm{(NR)}$'%('{'+label_mode+'n0}'))

        sp.plot(d_arctan[:,0], d_arctan[:,1], 'deepskyblue', linewidth=3.5, label = r'$\dot{\theta}_%s$'%('{'+label_mode+'}'))
        sp.legend(bbox_to_anchor=(0.98, 0.02), loc='lower right', fontsize = font_size, fancybox = True, framealpha = 1)
        plt.savefig(f'figs/darctan_{label_mode}.pdf', bbox_inches="tight")
