import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

from FitOvertone import FitFunctions
# from fit_functions import *
def plot_mismatch(mismatch, mins):
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['figure.figsize'] = [12, 8]  # plot image size

    font_size = 25
    lw = 3
    plt.close('all')
    
    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)
    plt.margins(0.05)
    plt.xlabel(r'$t_0 - t_{\mathrm{peak}} [M]$', fontsize = font_size)
    plt.ylabel(r'$\mathcal{M}$', fontsize = font_size)
    plt.xlim(0, mismatch['dtheta']['time'].values[-100])
    # mismatch
    colors = {'dtheta': 'black', 'strain': 'red'}
    labels = {'dtheta': r'$\dot{\theta}_{22}$ fit', 'strain': r'$\psi_{22}$ fit'}
    for key in ['dtheta', 'strain']:
        plt.plot(
            mismatch[key]['time'][:-100],
            mismatch[key]['mismatch'][:-100],
            color = colors[key],
            label = labels[key],
            lw = lw,
            )
        plt.plot(
            mismatch['energy_'+key]['time'][:-100],
            mismatch['energy_'+key]['mismatch'][:-100],
            color = colors[key],
            lw = lw,
            ls = '--'
            )

    # chosen initial times lines
    plt.axvline(
        x=mins['strain']['time'].values,
        color = colors['strain'],
        ls=':',
        lw=1
        )
    plt.axvline(
        x=mins['energy_dtheta']['time'].values,
        color = colors['dtheta'],
        ls=':',
        lw=1
        )

    # chosen first min
    plt.scatter(
        mins['strain']['time'], 
        mins['strain']['mismatch'],
        color = 'blue',
        marker = 'X',
        s=150,
        zorder = 10
        )
    plt.scatter(
        mins['energy_dtheta']['time'], 
        mins['energy_dtheta']['mismatch'],
        color = 'blue',
        marker = 'X',
        s=150,
        zorder = 10
        )

    # not considered first min
    plt.scatter(
        mins['energy_strain']['time'], 
        mins['energy_strain']['mismatch'],
        color = 'gray',
        marker = 'o',
        s=150,
        zorder = 10
        )
    plt.scatter(
        mins['dtheta']['time'], 
        mins['dtheta']['mismatch'],
        color = 'gray',
        marker = 'o',
        s=150,
        zorder = 10
        )

    plt.yscale('log')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 0.8), loc='upper right', fontsize = font_size, fancybox = True)
    plt.savefig('figs/mismatch.pdf', bbox_inches="tight")

def find_best_fit(mins, fit_pars):
    best_pars = {
        'dtheta': fit_pars['dtheta'].loc[mins['energy_dtheta']['time'].index],
        'strain': fit_pars['strain'].loc[mins['strain']['time'].index],
    }
    for (key, value) in best_pars.items():
        with open(f'data/overtone_fits/best_fit_{key}.json', 'w') as fp:
            json.dump(value.to_dict(orient='records'), fp, indent=4)
    return best_pars    

def find_pars_time_dtheta(data_dtheta, best_pars, omegas):
    R_t = best_pars['dtheta']['R'].values[0]*np.exp(
        (omegas['(2,2,1)']['omega_i'] - omegas['(2,2,0)']['omega_i'])*(
            best_pars['dtheta']['time'].values[0] - data_dtheta['time']
            )
        )
    dphi_t = (best_pars['dtheta']['dphi'].values[0] +
        (omegas['(2,2,0)']['omega_r']- omegas['(2,2,1)']['omega_r'])*(
            best_pars['dtheta']['time'].values[0] - data_dtheta['time']
        ))
    expected_pars = pd.DataFrame(np.array((
        data_dtheta['time'].values,
        R_t,
        dphi_t,
        )).T,
        columns=['time', 'R', 'dphi']
        )
    return expected_pars

def find_pars_time_strain(data_strain, best_pars, omegas):
    A_0 = best_pars['strain']['A_0'].values[0]*np.exp(
        (omegas['(2,2,0)']['omega_i'])*(
            best_pars['strain']['time'].values[0] - data_strain['time']
            )
        )

    A_1 = best_pars['strain']['A_1'].values[0]*np.exp(
        (omegas['(2,2,1)']['omega_i'])*(
            best_pars['strain']['time'].values[0] - data_strain['time']
            )
        )

    R_t = A_1/A_0

    phi_0 = (best_pars['strain']['phi_0'].values[0] +
        omegas['(2,2,0)']['omega_r']*(
            best_pars['strain']['time'].values[0] - data_strain['time']
        ))

    phi_1 = (best_pars['strain']['phi_1'].values[0] +
        omegas['(2,2,1)']['omega_r']*(
            best_pars['strain']['time'].values[0] - data_strain['time']
        ))

    dphi = phi_0 - phi_1

    expected_pars = pd.DataFrame(np.array((
        data_strain['time'].values,
        R_t,
        dphi,
        A_0,
        A_1,
        phi_0,
        phi_1
        )).T,
        columns=[
            'time',
            'R',
            'dphi',
            'A_0',
            'A_1',
            'phi_0',
            'phi_1',
            ]
        )
    return expected_pars

def plot_fitted_expected(fit_pars, best_pars, exp_pars, t_fin):
    # ploted fitted parameters and expected parameters and its relative errors as a function of time

    font_size = 20

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['figure.figsize'] = [14, 8]  # plot image size
    mpl.rcParams['ytick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    plt.close('all')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
    plt.margins(0.1)
    ax1.set_ylabel(r'$R = A_{221}/A_{220}$', fontsize=font_size)
    ax1.set_ylim(
        0,
        max(
            max(exp_pars['strain']['R']),
            max(exp_pars['dtheta']['R']),
            )
        )
    ax1.plot(fit_pars['strain']['time'], fit_pars['strain']['R'], color='red', label=r'$R^\mathrm{II}$', lw=3)
    ax1.plot(fit_pars['dtheta']['time'], fit_pars['dtheta']['R'], color='black', ls='--', label=r'$R^\mathrm{I}$', lw=3)

    ax1.axvline(best_pars['strain']['time'].values[0], color = 'red', ls=':', lw=1)
    ax1.axvline(best_pars['dtheta']['time'].values[0], color = 'black', ls=':', lw=1)

    ax1.plot(exp_pars['dtheta']['time'], exp_pars['dtheta']['R'], color='black', lw=2, ls=':', alpha=0.8,label=r"$\mathcal{R}^{\mathrm{I}}(t)$")
    ax1.plot(exp_pars['strain']['time'], exp_pars['strain']['R'], color='red', lw=2, ls=':', label=r"$\mathcal{R}^{\mathrm{II}}(t)$")
    ax1.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', fontsize=font_size, fancybox=True, framealpha=.2)

    ax3.set_xlim(0, t_fin)
    ax3.set_xlabel(r'$t_0 - t_{\mathrm{peak}} [M]$', fontsize=font_size)
    ax3.set_ylabel(r'relative difference', fontsize=font_size)
    ax3.set_yscale('log')
    ax3.set_ylim(pow(10, -4), pow(10, 0))
    ax3.plot(
        fit_pars['strain']['time'][fit_pars['strain']['time']<t_fin],(
            abs(fit_pars['strain']['R'][fit_pars['strain']['time']<t_fin] - 
                exp_pars['strain']['R'][exp_pars['strain']['time']<t_fin]) / 
            (fit_pars['strain']['R'])[fit_pars['strain']['time']<t_fin]),
            color='red', label=r'$\psi_{22}$ fit', lw=3)
    ax3.plot(fit_pars['dtheta']['time'][fit_pars['dtheta']['time']<t_fin],(
                abs(fit_pars['dtheta']['R'][fit_pars['dtheta']['time']<t_fin] - 
                    exp_pars['dtheta']['R'][exp_pars['dtheta']['time']<t_fin]) / 
                fit_pars['dtheta']['R'][fit_pars['dtheta']['time']<t_fin]),
            color='black', ls='--', label=r'$\dot{\theta}_{22}$ fit', lw=3)

    ax3.axvline(best_pars['strain']['time'].values[0], color = 'red', ls=':', lw=1)
    ax3.axvline(best_pars['dtheta']['time'].values[0], color = 'black', ls=':', lw=1)

    ax2.set_ylim(
        min(
            min(fit_pars['strain']['dphi'][fit_pars['strain']['time']<t_fin]),
            min(fit_pars['dtheta']['dphi'][fit_pars['dtheta']['time']<t_fin]),
            min(exp_pars['strain']['dphi']),
            min(exp_pars['dtheta']['dphi']),
            )*0.9,
        max(
            max(fit_pars['strain']['dphi'][fit_pars['strain']['time']<t_fin]),
            max(fit_pars['dtheta']['dphi'][fit_pars['dtheta']['time']<t_fin]),
            max(exp_pars['strain']['dphi']),
            max(exp_pars['dtheta']['dphi']),
            )*1.1,
    )

    ax2.set_ylabel(r'$\phi = \phi_{220} - \phi_{221} [\mathrm{rad}]$', fontsize=font_size)
    ax2.plot(fit_pars['strain']['time'], fit_pars['strain']['dphi'], color='red', label=r'$\phi^{\mathrm{II}}$', lw=3)
    ax2.plot(fit_pars['dtheta']['time'], fit_pars['dtheta']['dphi'], color='black', ls='--', label=r'$\phi^\mathrm{I}$',lw=3)

    ax2.plot(exp_pars['strain']['time'], exp_pars['strain']['dphi'], color='r', lw=2, ls=':', alpha=0.8, label=r"$\varphi^{\mathrm{II}}(t)$")
    ax2.plot(exp_pars['dtheta']['time'], exp_pars['dtheta']['dphi'], color='k', lw=2, ls=':', alpha=0.8, label=r"$\varphi^{\mathrm{I}}(t)$")

    ax2.axvline(best_pars['strain']['time'].values[0], color = 'red', ls=':', lw=1)
    ax2.axvline(best_pars['dtheta']['time'].values[0], color = 'black', ls=':', lw=1)

    ax2.legend(bbox_to_anchor=(0.01, 0.01), loc='lower left', fontsize=font_size, fancybox=True, framealpha=.2)

    ax4.set_xlim(0, t_fin)
    ax4.set_ylim(10 ** (-3), 10 ** (1))
    ax4.set_xlabel(r'$t_0 - t_{\mathrm{peak}} [M]$', fontsize=font_size)
    ax4.set_yscale('log')
    ax4.set_ylim(pow(10, -4), pow(10, 0))
    ax4.plot(
        fit_pars['strain']['time'][fit_pars['strain']['time']<t_fin],(
            abs(fit_pars['strain']['dphi'][fit_pars['strain']['time']<t_fin] - 
                exp_pars['strain']['dphi'][exp_pars['strain']['time']<t_fin]) / 
            (fit_pars['strain']['dphi'])[fit_pars['strain']['time']<t_fin]),
            color='red', lw=3)
    ax4.plot(fit_pars['dtheta']['time'][fit_pars['dtheta']['time']<t_fin],(
                abs(fit_pars['dtheta']['dphi'][fit_pars['dtheta']['time']<t_fin] - 
                    exp_pars['dtheta']['dphi'][exp_pars['dtheta']['time']<t_fin]) / 
                fit_pars['dtheta']['dphi'][fit_pars['dtheta']['time']<t_fin]),
            color='black', ls='--', lw=3)

    ax4.axvline(best_pars['strain']['time'].values[0], color = 'red', ls=':', lw=1)
    ax4.axvline(best_pars['dtheta']['time'].values[0], color = 'black', ls=':', lw=1)

    plt.subplots_adjust(hspace=.001)
    fig.tight_layout()
    fig.savefig('figs/fit_pars.pdf', bbox_inches="tight")

def plot_best_fits(best_pars, data_dtheta, data_strain, omegas, modes, t_initial):
    # plot best fit

    # compute best fits as a function of time
    strain_n1_fitted = abs(
        f_Re_2_modes(
            data_strain['time'] - best_pars['strain']['time'].values[0],
            best_pars['strain']['A_0'].values[0],
            best_pars['strain']['A_1'].values[0],
            best_pars['strain']['phi_0'].values[0],
            best_pars['strain']['phi_1'].values[0],
            omegas[modes[0]]['omega_r'],
            omegas[modes[1]]['omega_r'],
            omegas[modes[0]]['omega_i'],
            omegas[modes[1]]['omega_i'],
            )
        + 1j*f_Im_2_modes(
            data_strain['time'] - best_pars['strain']['time'].values[0],
            best_pars['strain']['A_0'].values[0],
            best_pars['strain']['A_1'].values[0],
            best_pars['strain']['phi_0'].values[0],
            best_pars['strain']['phi_1'].values[0],
            omegas[modes[0]]['omega_r'],
            omegas[modes[1]]['omega_r'],
            omegas[modes[0]]['omega_i'],
            omegas[modes[1]]['omega_i'],
            )
        )

    dtheta_fitted = d_theta(
        data_dtheta['time'] - best_pars['dtheta']['time'].values[0],
        best_pars['dtheta']['R'].values[0],
        best_pars['dtheta']['dphi'].values[0],
        omegas[modes[0]]['omega_r'],
        omegas[modes[1]]['omega_r'],
        omegas[modes[0]]['omega_i'],
        omegas[modes[1]]['omega_i'],
        )

    strain_n0_fitted = abs(
        f_Re_1_mode(
            data_strain['time'] - best_pars['strain']['time'].values[0],
            best_pars['strain']['A_0'].values[0],
            best_pars['strain']['phi_0'].values[0],
            omegas[modes[0]]['omega_r'],
            omegas[modes[0]]['omega_i'],
            )
        + 1j*f_Im_1_mode(
            data_strain['time'] - best_pars['strain']['time'].values[0],
            best_pars['strain']['A_0'].values[0],
            best_pars['strain']['phi_0'].values[0],
            omegas[modes[0]]['omega_r'],
            omegas[modes[0]]['omega_i'],
            )
        )

    # plot fits and residuals
    font_size = 20

    plt.rcParams['figure.figsize'] = [14, 8]  # plot image size
    mpl.rcParams['ytick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    plt.close('all')

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
    plt.margins(0.1)
    lw = 3
    # strain fits
    ax1.set_ylabel(r'$\left|\psi_{22}\right|$', fontsize=font_size)
    ax1.set_xlim([0, 60])
    ax1.plot(data_strain['time'], abs(data_strain['real'] + 1j * data_strain['imag']), color='deepskyblue', lw=lw, label=r'NR')
    ax1.plot(data_strain['time'], strain_n1_fitted, ls='-', color='red', lw=lw, label=r'$n = 0 + 1$')
    ax1.plot(data_strain['time'], strain_n0_fitted, ls='--', color='limegreen', lw=lw, label=r'$n = 0$')
    ax1.set_ylim([0, 0.5])
    ax1.legend(bbox_to_anchor=(1, 0.98), loc='upper right', fontsize=font_size, fancybox=True, framealpha=1)
    ax1.axvline(t_initial, ls=':', color='limegreen')
    ax1.axvline(best_pars['strain']['time'].values[0], color = 'red', ls=':', lw=1)
    

    # strain relative difference
    ax2.set_ylabel(r'relative difference', fontsize=font_size)
    ax2.set_xlabel(r'$t - t_{\mathrm{peak}} [M]$', fontsize=font_size)
    ax2.set_yscale('log')
    ax2.set_xlim([0, 60])
    ax2.set_ylim(bottom=pow(10, -5), top=pow(10, 0))
    ax2.plot(data_strain['time'], abs(abs(data_strain['real'] + 1j * data_strain['imag']) - strain_n0_fitted)/
             abs(data_strain['real'] + 1j * data_strain['imag']), color='limegreen', lw=lw, ls='--')
    ax2.plot(data_strain['time'], abs(abs(data_strain['real'] + 1j * data_strain['imag']) - strain_n1_fitted)/
             abs(data_strain['real'] + 1j * data_strain['imag']), color='red', lw=lw)
    ax2.axvline(t_initial, ls=':', color='limegreen')
    ax2.axvline(best_pars['strain']['time'].values[0], color = 'red', ls=':', lw=1)

    # dtheta fits
    ax3.set_ylabel(r'$\dot{\theta}_{22} [1/M]$', fontsize=font_size)
    ax3.set_xlim([0, 60])
    ax3.plot(data_dtheta['time'], data_dtheta['dtheta'], color='deepskyblue', lw=lw, label=r'NR')
    ax3.axhline(y=omegas[modes[0]]['omega_r'], color='limegreen', linewidth=lw, ls='--', label=r'$n = 0$')
    ax3.plot(data_dtheta['time'], dtheta_fitted, ls='--', color='k', lw=lw, label=r'$n = 0 + 1$')
    ax3.axvline(best_pars['dtheta']['time'].values[0], color = 'black', ls=':', lw=1)
    ax3.axvline(t_initial, ls=':', color='limegreen')
    ax3.legend(bbox_to_anchor=(1, 0.02), loc='lower right', fontsize=font_size, fancybox=True, framealpha=1)

    # dtheta relative differences
    ax4.set_ylim(bottom=pow(10, -5), top=pow(10, 0))
    ax4.set_xlabel(r'$t - t_{\mathrm{peak}} [M]$', fontsize=font_size)
    ax4.set_xlim([0, 60])
    ax4.set_yscale('log')
    ax4.plot(data_dtheta['time'], abs(data_dtheta['dtheta'] - omegas[modes[0]]['omega_r'])/data_dtheta['dtheta'], color='limegreen', lw=lw, ls='--')
    ax4.plot(data_dtheta['time'], abs(data_dtheta['dtheta'] - dtheta_fitted)/data_dtheta['dtheta'], color='k', lw=lw, ls='--')
    ax4.axvline(best_pars['dtheta']['time'].values[0], color = 'black', ls=':', lw=1)
    ax4.axvline(t_initial, ls=':', color='limegreen')

    plt.subplots_adjust(hspace=.001)
    fig.tight_layout()
    fig.savefig('figs/best_fits.pdf', bbox_inches="tight")

def main(omegas):
    
    #import data
    t_initial = np.genfromtxt('data/times_fundamental.dat')[0]

    data_strain = np.genfromtxt(f'data/waveforms/peak_22_l2m2.dat', delimiter='\t')
    data_dtheta = np.genfromtxt(f'data/d_theta/peak_22_l2m2.dat', delimiter='\t')

    # transform data to dataframe
    data_strain = pd.DataFrame(data_strain, columns = ['time', 'real', 'imag'])
    data_dtheta = pd.DataFrame(data_dtheta, columns = ['time', 'dtheta'])
    modes = ['(2,2,0)', '(2,2,1)']

    mismatch = {}
    mins = {}
    fit_pars = {}
    columns_fit = {
        'dtheta': ['time', 'R', 'dphi'],
        'strain': ['time', 'A_0', 'A_1', 'phi_0', 'phi_1', 'R', 'dphi']
        }

    for key in ['dtheta', 'energy_dtheta', 'strain', 'energy_strain']:
        mismatch[key] = np.genfromtxt(f'data/overtone_fits/mismatch_{key}.dat')
        mismatch[key] = pd.DataFrame(mismatch[key], columns = ('time', 'mismatch'))

        aux = mismatch[key]['mismatch'][0]
        i = 1
        if key == 'energy_strain':
            i = mins['strain']['time'].index[0]-20
        while mismatch[key]['mismatch'][i] < aux:
            aux = mismatch[key]['mismatch'][i]
            i += 1
                
        mins[key] = {
            'time': mismatch[key]['time'][mismatch[key]['time']==mismatch[key]['time'][i]],
            'mismatch': mismatch[key]['mismatch'][
                        mismatch[key]['time']==mismatch[key]['time'][i]]
        }
        try:
            fit_pars[key] = np.genfromtxt(f'data/overtone_fits/fitpar_{key}.dat')
            fit_pars[key] = pd.DataFrame(fit_pars[key], columns = columns_fit[key])

        except: pass
    

    # plot mismatch
    plot_mismatch(mismatch, mins)

    best_pars = find_best_fit(mins, fit_pars)
    exp_pars = {}
    exp_pars['dtheta'] = find_pars_time_dtheta(data_dtheta, best_pars, omegas)
    exp_pars['strain'] = find_pars_time_strain(data_strain, best_pars, omegas)

    for (key, value) in exp_pars.items():
        with open(f'data/overtone_fits/best_fit_{key}_t10.json', 'w') as fp:
            json.dump(value[round(value['time'],1) == 10].to_dict(orient='records'), fp, indent=4)

    initial_times = []
    for (key, values) in mins.items():
        initial_times.append(values['time'].values)
    print(max(initial_times) + 10)
    plot_fitted_expected(fit_pars, best_pars, exp_pars, max(initial_times)[0] + 10)
    plot_best_fits(best_pars, data_dtheta, data_strain, omegas, modes, t_initial)

