import json
import os

import numpy as np
import pandas as pd

from numpy import abs, exp, cos, sin
from lmfit import Parameters #fitting library
import lmfit
from numpy import arctan, pi
import math
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

class FundamentalFit():

    def __init__(
        self,
        file_path:str,
        lm_modes:list,
        lm_dominant:list,
        ):
        self.a_M, self.omegas_fit = self.fit_fundamental_mode(file_path, lm_modes, lm_dominant)
        self.fit_fundamental_mode_amplitude(file_path, lm_modes, lm_dominant)

    def fit_fundamental_mode(
        self,
        file_path:str,
        lm_modes:list,
        lm_dominant:str,
        ):

        # compute initial and final time from lm_dominant mode
        t_initial, t_final = self._find_times_fundamental(lm_dominant, lm_dominant)
        
        # create folder data/fits
        if not os.path.exists('data/fits'):
            os.makedirs('data/fits')

        # import simulation frequencies
        with open('data/omegas/Mi_qnm_freqs_sim.json') as file:
            omegas = json.loads(file.read())

        # import simulation info
        with open(file_path+'/metadata.json') as f:
            par_simu = json.load(f)

        # find fundamental mode frequencies
        # fitting datas
        omegas_fit = {}
        label_dominant = f'l{lm_dominant[0]}m{lm_dominant[1]}'
        for (l,m) in lm_modes:
            key = f'l{l}m{m}'
            data_peak = np.genfromtxt(f'data/waveforms/peak_{label_dominant}_{key}.dat', delimiter='\t')
            # transform data to dataframe
            data_peak = pd.DataFrame(data_peak, columns = ['time', 'real', 'imag'])
            # select data in the interval [t_initial, t_final]
            data = data_peak[(data_peak['time']>=t_initial)&(data_peak['time']<=t_final)]
            # ! for SXSBBH0305 the fit will be more similar to the simulation values when t_final -= 23
            
            # selec mode label
            mode = f'({l},{m},0)'
            omegas_fit[mode] = {}

            # fitting parameters
            params = Parameters()
            params.add('wr', value=omegas[mode]['omega_r'], vary=True)
            params.add('wi', value=omegas[mode]['omega_i'], vary=True)
            params.add('A0r', value=0.01, vary=True)
            params.add('t0r', value=0.01, vary=True, min = -2*np.pi, max = 2*np.pi)
            params.add('A0i', expr = 'A0r')
            params.add('n0', value = 0, min = -4, max = 4)
            # params.add('t0i', expr = 't0r + pi*n0')

            # fit
            fit = lmfit.minimize(
                self._residual,
                params,
                kws={
                    "x": data['time'] - t_initial,
                    "datRe": data['real'],
                    "datIm": data['imag']
                    },
                method = 'least_squares',
                xtol=1e-15
                )

            # check if phase fit is consistent
            if np.abs(fit.params['n0'].value) < np.abs(round(fit.params['n0'].value,0)) - 0.1 \
                    or np.abs(fit.params['n0'].value) > np.abs(round(fit.params['n0'].value,0)) + 0.1:
                sys.exit("Bad phase fit, aborting")

            # save fitted parameters to file
            orig_stdout = sys.stdout
            fitfile = open(f'data/fits/fit_results_fundamental_{key}.txt', 'w')
            sys.stdout = fitfile
            print(lmfit.fit_report(fit))
            sys.stdout = orig_stdout
            fitfile.close()
            del fitfile

            omegas_fit[mode]['omega_r'] = fit.params['wr'].value
            omegas_fit[mode]['omega_i'] = fit.params['wi'].value
            
            self.plot_fit(data_peak, key, fit, t_initial, t_final)
            
        # import fitting coefficients
        fit_coeffs = self._import_fit_coeffs_a_M(file_path)
        lm_index = f'({lm_dominant[0]},{lm_dominant[1]},0)'
        q1,q2,q3 = fit_coeffs.loc[lm_index].values[3:]
        a_M = self.aoverM(
            omegas_fit[lm_index]['omega_r']*par_simu['remnant_mass'],
            omegas_fit[lm_index]['omega_i']*par_simu['remnant_mass'],
            q1,
            q2,
            q3,
            )

        return a_M, omegas_fit

    def fit_fundamental_mode_amplitude(
        self,
        file_path:str,
        lm_modes:list,
        lm_dominant:str,
        ):

        # compute initial and final time from lm_dominant mode
        t_initial, t_final = self._find_times_fundamental(lm_dominant, lm_dominant)
        
        # create folder data/fits
        if not os.path.exists('data/fits'):
            os.makedirs('data/fits')

        # import simulation frequencies
        with open('data/omegas/Mi_qnm_freqs_sim.json') as file:
            omegas = json.loads(file.read())

        # import simulation info
        with open(file_path+'/metadata.json') as f:
            par_simu = json.load(f)

        # find fundamental mode frequencies
        # fitting datas
        omegas_fit = {}
        label_dominant = f'l{lm_dominant[0]}m{lm_dominant[1]}'
        for (l,m) in lm_modes:
            key = f'l{l}m{m}'
            data_peak = np.genfromtxt(f'data/waveforms/peak_{label_dominant}_{key}.dat', delimiter='\t')
            # transform data to dataframe
            data_peak = pd.DataFrame(data_peak, columns = ['time', 'real', 'imag'])
            # select data in the interval [t_initial, t_final]
            data = data_peak[(data_peak['time']>=t_initial)&(data_peak['time']<=t_final)]
            
            # selec mode label
            mode = f'({l},{m},0)'
            omegas_fit[mode] = {}

            # fitting parameters
            params = Parameters()
            params.add('wr', value=omegas[mode]['omega_r'], vary=False)
            params.add('wi', value=omegas[mode]['omega_i'], vary=False)
            params.add('A0r', value=0.01, vary=True)
            params.add('t0r', value=0.01, vary=True, min = -2*np.pi, max = 2*np.pi)
            params.add('A0i', expr = 'A0r')
            params.add('n0', value = 0, min = -4, max = 4)
            # params.add('t0i', expr = 't0r + pi*n0')

            # fit
            fit = lmfit.minimize(
                self._residual,
                params,
                kws={
                    "x": data['time'] - t_initial,
                    "datRe": data['real'],
                    "datIm": data['imag']
                    },
                method = 'least_squares',
                xtol=1e-15
                )

            # check if phase fit is consistent
            if np.abs(fit.params['n0'].value) < np.abs(round(fit.params['n0'].value,0)) - 0.1 \
                    or np.abs(fit.params['n0'].value) > np.abs(round(fit.params['n0'].value,0)) + 0.1:
                sys.exit("Bad phase fit, aborting")

            # save fitted parameters to file
            orig_stdout = sys.stdout
            fitfile = open(f'data/fits/fit_results_fundamental_amplitudes_{key}.txt', 'w')
            sys.stdout = fitfile
            print(lmfit.fit_report(fit))
            sys.stdout = orig_stdout
            fitfile.close()
            del fitfile

            with open(f'data/fits/fit_amp_phase_fundamental_{key}.dat', 'w') as f:
                np.savetxt(
                    f,
                    [
                        data['time'].values[0],
                        fit.params['A0r'].value,
                        fit.params['t0r'].value
                    ],
                    delimiter='\t',
                    header = '#(1)inital time t0 (2)A_0 at t0 (3)phi_0 at t0'
                    )

            with open(f'data/fits/best_fit_t10_{key}.json', 'w') as fp:
                json.dump({
                    'time': 10,
                    'A': fit.params['A0r'].value*np.exp(
                        (omegas[mode]['omega_i'])*(
                            data['time'].values[0] - 10
                            )),
                    'phi': (fit.params['t0r'].value +
                        omegas[mode]['omega_r']*(
                            data['time'].values[0] - 10
                        ))
                    }, fp, indent=4)

            
            self.plot_fit(data_peak, key+"_omega_sim", fit, t_initial, t_final)

    def aoverM(self, wr, wi, q1, q2, q3):
        # Fits to Kerr QNM frequencies:	https://pages.jh.edu/~eberti2/ringdown/
        return 1 - ((wr/(2*wi) - q1)/q2)**(1/q3)

    def _find_times_fundamental(
        self,
        lm_times:list,
        lm_dominant:str,
        var=0.01,
        ):
        """Find the interval where the derivative of the complex
        phase of (l,m) mode is constant. Here we consider constant
        if the derivative of the phases varies less var%. This is 
        equivalent as the interval where the waveform is well 
        described by the fundamental mode (l,m,0)

        Parameters
        ----------
        lm_times : list
            (l,m) mode used to compute de initial and final time. 
            Should be a list such that lm[0] = l and lm[1] = m
        
        lm_dominant : list
            (l,m) of the considered dominant mode. This is the mode
            with which the peak time is considered. 
            Should be a list such that lm[0] = l and lm[1] = m
        
        var: float
            Percentage which the derivative of theta can vary to
            be considered constanta. By default: 0.01.

        Returns
        -------
        tuple
            Returns initial and final times of the fundamental mode
        """
        # find interval where derivative of theta of (l,m) is constant
        # this is equivalent as the interval where the waveform is
        # well described by the fundamental mode (l,m,0)
        label_d = f'l{lm_dominant[0]}m{lm_dominant[1]}'
        label_t = f'l{lm_times[0]}m{lm_times[1]}'
        d_arctan = np.genfromtxt(f'data/d_theta/peak_{label_d}_{label_t}.dat', delimiter='\t')
        const = []
        for i in range(len(d_arctan)):
            k = i
            j = i
            while d_arctan[i][1]  <= d_arctan[j][1]*(1+var) and d_arctan[i][1]  >= d_arctan[j][1]*(1-var):
                k = j
                j += 1
                if j >= len(d_arctan):
                    break
            aux = d_arctan[i:k+1]
            if len(aux) > len(const):
                const = aux

        t_initial = const[0][0]
        t_final = const[-1][0]

        np.savetxt('data/times_fundamental.dat', (t_initial, t_final), delimiter='\t', header = f'#(0) initial (1) final # ({label_t[0]},{label_t[1]},0) mode')

        return t_initial, t_final

    def _import_fit_coeffs_a_M(
        self,
        file_path:str,
        ):
        """Import fit coefficients for a and M as a function 
        of omega_r and omega_i. Original data from 
        https://pages.jh.edu/~eberti2/ringdown/fitcoeffsWEB.dat

        Parameters
        ----------
        file_path : string
            Path containing fitcoeffs.dat

        Returns
        -------
        pandas dataframe
            Returns a dataframe of the fitting coefficients. Rows
            indexes are (l,m,n) and each collumn is a fitting
            coefficient.
        """
        file = np.genfromtxt(file_path+'/fitcoeffs.dat')
        coeff_cols = np.arange(3,len(file[0]))
        qnm_indexes = [f'({int(l)},{int(m)},{int(n)})' for (l,m,n) in file[:,[0,1,2]]]
        df = pd.DataFrame(
            file[:,coeff_cols],
            columns = ('f1','f2','f3','q1','q2','q3'),
            index = qnm_indexes,
            )
        return df

    # fitting functions
    def _fRe_0(self, x, A_0r, t_0r, w_r, w_i):
        """Computes a single QNM real part waveform.

        Parameters
        ----------
        x : array_like
            time array.
        A_0r : float
            QNM amplitude.
        t_0r : float
            QNM phase.
        w_r : float
            QNM real frequency.
        w_i : float
            QNM imaginary frequency.

        Returns
        -------
        array
            Returns QNM waveform.
        """
        return A_0r*np.exp(-w_i*x)*np.cos(w_r*x-t_0r)

    def _fIm_0(self, x, A_0i, t_0i, w_r, w_i):
        """Computes a single QNM imaginary part waveform.

        Parameters
        ----------
        x : array_like
            time array.
        A_0r : float
            QNM amplitude.
        t_0r : float
            QNM phase.
        w_r : float
            QNM real frequency.
        w_i : float
            QNM imaginary frequency.

        Returns
        -------
        array
            Returns QNM waveform.
        """
        return A_0i*np.exp(-w_i*x)*np.sin(w_r*x-t_0i)

    def _residual(self, params,  x=None, datRe=None, datIm=None):
        """Computes residual of data and fitted function.

        Parameters
        ----------
        params : dict
            dictionary of QNM parameters
        x : list,   
            time array data.
        datRe : array,
            waveform real part data.
        datIm : array,
            waveform imaginary part data.

        Returns
        -------
        tuple
            Returns residual of fitted waveform and data.
        """
        modelRe = self._fRe_0(x, params['A0r'], params['t0r'], params['wr'], params['wi'])
        modelIm = self._fIm_0(x, params['A0i'], params['n0']*np.pi + params['t0r'], params['wr'], params['wi'])

        residRe = datRe - modelRe
        residIm = datIm - modelIm
        return np.concatenate((residRe, residIm))

    def plot_fit(self, data, label_mode, fit, t_initial, t_final):
        # plot fitted strain
        font_size = 30
        lw = 3


        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['figure.figsize'] = [12, 8]  # plot image size
        mpl.rcParams['ytick.labelsize'] = font_size
        mpl.rcParams['xtick.labelsize'] = font_size

        plt.close('all')
        ylabel = '{'+label_mode[1]+label_mode[3]+'}'
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        plt.margins(x=0.01, y=0.01)
        ax2.set_xlabel(r'$t - t_{\mathrm{peak}} [M]$', fontsize=font_size)
        ax1.set_ylabel(r'$\mathrm{Re}$'+r'$(\psi_{})$'.format(ylabel), fontsize=font_size)
        ax2.set_ylabel(r'$\mathrm{Im}$'+r'$(\psi_{})$'.format(ylabel), fontsize=font_size)
        # ax1.set_ylim(-0.6, 0.5)
        # ax2.set_ylim(-0.5, 0.6)
        ax1.plot(data['time'], data['real'], color='deepskyblue', lw=lw, label='NR')
        ax1.plot(
            data['time'],
            self._fRe_0(
                data['time'] - t_initial,
                fit.params['A0r'].value,
                fit.params['t0r'].value,
                fit.params['wr'].value,
                fit.params['wi'].value,
                ),
            ls='--',
            color='limegreen',
            lw=lw,
            label=r'$n = 0$ fit',
            )
        ax2.plot(data['time'], data['imag'], color='deepskyblue', lw=lw, label='NR')
        ax2.plot(
            data['time'],
            self._fIm_0(
                data['time'] - t_initial,
                fit.params['A0i'].value,
                fit.params['n0'].value*np.pi + fit.params['t0r'].value,
                fit.params['wr'].value,
                fit.params['wi'].value,
                ),
            ls='--',
            color='limegreen',
            lw=lw,
            label=r'$n = 0$ fit',
            )

        ax1.axvline(t_initial, ls=':', color='limegreen')
        ax1.axvline(t_final, ls=':', color='limegreen')
        ax2.axvline(t_initial, ls=':', color='limegreen')
        ax2.axvline(t_final, ls=':', color='limegreen')
        ax1.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=font_size - 5, fancybox=True, framealpha=1)
        fig.subplots_adjust(hspace=0)
        fig.savefig(f'figs/waveforms_fit_{label_mode}.pdf', bbox_inches="tight")


