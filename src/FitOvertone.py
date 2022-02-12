import json
import numpy as np
from numpy import pi, exp, sin, cos, sqrt
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from lmfit import Model, Parameters #fitting library
import lmfit

import matplotlib as mpl

class FitOvertone:

    def __init__(
        self,
        omegas,
        lm_fit=(2,2),
        lm_dominant=(2,2),
        ):
        self.plots = Plots()
        self.fitfun = FitFunctions()
        # create folders for data/overtone_fits
        if not os.path.exists('data/overtone_fits'):
            os.makedirs('data/overtone_fits')

        label_fit = f'l{lm_fit[0]}m{lm_fit[1]}'
        label_dominant = f'l{lm_dominant[0]}m{lm_dominant[1]}'

        data_dtheta, data_strain = self._import_data(label_fit, label_dominant)
        
        modes = (f'({lm_fit[0]},{lm_fit[1]},0)', f'({lm_fit[0]},{lm_fit[1]},1)')

        # fit and save parameters
        fit_dtheta = self.compute_fit_dtheta(data_dtheta, omegas, modes)
        self.save_dtheta_fit(data_dtheta, fit_dtheta)
        fit_strain = self.compute_fit_strain(data_strain, omegas, modes)
        self.save_strain_fit(data_strain, fit_strain)

        mis_dtheta = self.compute_mismatch_dtheta(data_dtheta, fit_dtheta, omegas, modes)
        mis_strain = self.compute_mismatch_strain(data_strain, fit_strain, omegas, modes)
        mismatch = {**mis_dtheta, **mis_strain}
        
        mins, fit_pars = self._compute_min_mismatch(mismatch)

        self.plots.plot_mismatch(mismatch, mins)

        best_pars = self.find_best_fit(mins, fit_pars)

        exp_pars = {}
        exp_pars['dtheta'] = self.find_pars_time_dtheta(data_dtheta, best_pars, omegas)
        exp_pars['strain'] = self.find_pars_time_strain(data_strain, best_pars, omegas)


        initial_times = []
        for values in mins.values():
            initial_times.append(values['time'])

        self.plots.plot_fitted_expected(fit_pars, best_pars, exp_pars, max(initial_times)[0] + 10)
        
        t_initial = np.genfromtxt('data/times_fundamental.dat')[0]
        self.plots.plot_best_fits(best_pars, data_dtheta, data_strain, omegas, modes, t_initial)

    def _import_data(
        self,
        label_fit:str,
        label_dominant:str,
        ):
        # import final time
        t_final = np.genfromtxt('data/times_fundamental.dat')[1]

        # import data
        data_strain = np.genfromtxt(f'data/waveforms/peak_{label_dominant}_{label_fit}.dat', delimiter='\t')
        data_dtheta = np.genfromtxt(f'data/d_theta/peak_{label_dominant}_{label_fit}.dat', delimiter='\t')

        # transform data to dataframe
        data_strain = pd.DataFrame(data_strain, columns = ['time', 'real', 'imag'])
        data_dtheta = pd.DataFrame(data_dtheta, columns = ['time', 'dtheta'])

        # select data in the interval [0, t_final]
        data_strain = data_strain[data_strain['time']<=t_final]
        
        # remove final point in fit_overtone with high error
        t_final_dtheta = t_final - 5
        data_dtheta = data_dtheta[data_dtheta['time']<=t_final_dtheta]

        return data_dtheta, data_strain

    def compute_fit_dtheta(self, data_dtheta, omegas, modes):
        # fit R and phi for theta dot at all times

        # stop fitting before last point
        lastfit = 10

        fit_dtheta = []
        ## define fit parameters
        params_dtheta = Parameters()
        params_dtheta.add('R', value = 0.01)
        params_dtheta.add('phi', value=1, min=0, max=2 * np.pi)

        ## fit at t = t_peak
        fit_dtheta.append(
            lmfit.minimize(
                self.fitfun.residual_d_theta,
                params_dtheta,
                method='least_squares',
                kws={
                    'wr0': omegas[modes[0]]['omega_r'],
                    'wr1': omegas[modes[1]]['omega_r'],
                    'wi0': omegas[modes[0]]['omega_i'],
                    'wi1': omegas[modes[1]]['omega_i'],
                    "x": data_dtheta['time'].values, 
                    "dat_d_theta": data_dtheta['dtheta'].values,
                    },
                xtol=3e-16,
                ftol=3e-16,
                )
            )

        ## fit at t > t_peak
        for i in tqdm(range(1,len(data_dtheta)- lastfit)):
            fit_dtheta.append(
                lmfit.minimize(
                    self.fitfun.residual_d_theta,
                    params_dtheta,
                    method='least_squares',
                    kws={
                        'wr0': omegas[modes[0]]['omega_r'],
                        'wr1': omegas[modes[1]]['omega_r'],
                        'wi0': omegas[modes[0]]['omega_i'],
                        'wi1': omegas[modes[1]]['omega_i'],
                        "x": data_dtheta['time'][i:].values - data_dtheta['time'][i],
                        "dat_d_theta": data_dtheta['dtheta'][i:].values,
                        },
                    xtol=3e-16,
                    ftol=3e-16,
                    )
                )
        return fit_dtheta

    def save_dtheta_fit(self, data_dtheta, fit_dtheta):
        save_dtheta = []
        for i in range(len(fit_dtheta)):
            save_dtheta.append([data_dtheta['time'][i], fit_dtheta[i].params['R'].value, fit_dtheta[i].params['phi'].value])

        # -R ->R and phi in [0,2pi], see corrigir_fase.pdf
        for i in range(len(save_dtheta)):
            if save_dtheta[i][1]<0: # if R < 0, add (2n + 1)pi to change the sign
                save_dtheta[i][1] = -save_dtheta[i][1]
                if save_dtheta[i][2] < - 3*np.pi:
                    save_dtheta[i][2] = save_dtheta[i][2]  + 5*np.pi
                elif save_dtheta[i][2] < -np.pi:
                    save_dtheta[i][2] = save_dtheta[i][2] + 3*np.pi
                elif save_dtheta[i][2] < 0:
                    save_dtheta[i][2] = save_dtheta[i][2] + np.pi
                elif save_dtheta[i][2] > 3*np.pi:
                    save_dtheta[i][2] = save_dtheta[i][2] - 3*np.pi
                else:
                    save_dtheta[i][2] = save_dtheta[i][2] - np.pi
            # adding 2npi does not change the wave: this limits phi in [0,2pi]
            while save_dtheta[i][2] < 0:
                save_dtheta[i][2] = save_dtheta[i][2] + 2*np.pi
            while save_dtheta[i][2] > 2*np.pi:
                save_dtheta[i][2] = save_dtheta[i][2] - 2*np.pi

        with open('data/overtone_fits/fitpar_dtheta.dat', 'w') as f:
            np.savetxt(f, save_dtheta,  delimiter='\t', header = '#(1)time (2)R (3)phi')

    def compute_fit_strain(self, data_strain, omegas, modes):
        # stop fitting before last point
        lastfit = 10
        # fundamental mode + 1st overtone
        fit_strain = []

        ## define fit parameters
        params_strain = Parameters()
        params_strain.add('A0r', value=0.01, vary=True)
        params_strain.add('A1r', value=0.01, vary=True)
        params_strain.add('t0r', value=0.01, vary=True, min=-2*np.pi, max=2 * np.pi)
        params_strain.add('t1r', value=0.01, vary=True, min=-2*np.pi, max=2 * np.pi)
        params_strain.add('A0i', expr='A0r')
        params_strain.add('A1i', expr='A1r')
        params_strain.add('n0', value=-1, min=-4, max=4)
        params_strain.add('n1', value=-1, min=-4, max=4)
        params_strain.add('t0i', expr='t0r + pi*n0')
        params_strain.add('t1i', expr='t1r + pi*n1')

        ## fit 
        for i in tqdm(range(len(data_strain)- lastfit)):
            fit_strain.append(
                lmfit.minimize(
                    self.fitfun.residual_2_modes,
                    params_strain,
                    method='least_squares',
                    kws={
                        'wr0': omegas[modes[0]]['omega_r'],
                        'wr1': omegas[modes[1]]['omega_r'],
                        'wi0': omegas[modes[0]]['omega_i'],
                        'wi1': omegas[modes[1]]['omega_i'],
                        "x": data_strain['time'][i:].values - data_strain['time'][i],
                        "datRe": data_strain['real'][i:].values,
                        "datIm": data_strain['imag'][i:].values,
                        },
                    xtol=3e-16,
                    ftol=3e-16
                    )
                )
        return fit_strain

    def save_strain_fit(self, data_strain, fit_strain):
        save_strain = []
        for i in range(len(fit_strain)):
            save_strain.append([
                data_strain['time'][i],
                fit_strain[i].params['A0r'].value,
                fit_strain[i].params['A1r'].value,
                fit_strain[i].params['t0i'].value,
                fit_strain[i].params['t1i'].value,
                (fit_strain[i].params['A1r'].value/
                    fit_strain[i].params['A0r'].value),
                (fit_strain[i].params['t0i'].value - 
                    fit_strain[i].params['t1i'].value),
            ])
            
        # -R ->R and phi in [0,2pi], see corrigir_fase.pdf
        for i in range(len(save_strain)):
            if save_strain[i][5]<0: # if R < 0, add (2n + 1)pi to change the sign
                save_strain[i][5] = -save_strain[i][5]
                if save_strain[i][6] < - 3*np.pi:
                    save_strain[i][6] = save_strain[i][6]  + 5*np.pi
                elif save_strain[i][6] < -np.pi:
                    save_strain[i][6] = save_strain[i][6] + 3*np.pi
                elif save_strain[i][6] < 0:
                    save_strain[i][6] = save_strain[i][6] + np.pi
                elif save_strain[i][6] > 3*np.pi:
                    save_strain[i][6] = save_strain[i][6] - 3*np.pi
                else:
                    save_strain[i][6] = save_strain[i][6] - np.pi
            # adding 2npi does not change the wave: this limits phi in [0,2pi]
            while save_strain[i][6] < 0:
                save_strain[i][6] = save_strain[i][6] + 2*np.pi
            while save_strain[i][6] > 2*np.pi:
                save_strain[i][6] = save_strain[i][6] - 2*np.pi

        with open('data/overtone_fits/fitpar_strain.dat', 'w') as f:
            np.savetxt(f, save_strain,  delimiter='\t', header = '#(1)time (2)A_0 (3)A_1 (4)phi_0 (5)phi_1 (6)R (7)dphi')

    def compute_mismatch_strain(self, data_strain, fit_strain, omegas, modes):

        Re_strain = [np.array(self.fitfun.f_Re_2_modes(
                    data_strain['time'][i:] - data_strain['time'][i],
                    fit_strain[i].params['A0r'].value,
                    fit_strain[i].params['A1r'].value,
                    fit_strain[i].params['t0r'].value,
                    fit_strain[i].params['t1r'].value,
                    omegas[modes[0]]['omega_r'],
                    omegas[modes[1]]['omega_r'],
                    omegas[modes[0]]['omega_i'],
                    omegas[modes[1]]['omega_i'],
                )) for i in range(len(fit_strain))]

        Im_strain = [np.array(self.fitfun.f_Im_2_modes(
                    data_strain['time'][i:] - data_strain['time'][i],
                    fit_strain[i].params['A0i'].value,
                    fit_strain[i].params['A1i'].value,
                    fit_strain[i].params['t0i'].value,
                    fit_strain[i].params['t1i'].value,
                    omegas[modes[0]]['omega_r'],
                    omegas[modes[1]]['omega_r'],
                    omegas[modes[0]]['omega_i'],
                    omegas[modes[1]]['omega_i'],
                )) for i in range(len(fit_strain))]

        mis_strain = [
            self.fitfun.mismatch_inner(
                data_strain['real'][i:].values + 1j*data_strain['imag'][i:].values,
                Re_strain[i] + 1j*Im_strain[i],
                data_strain['time'][i:].values,
            )
            for i in range(len(fit_strain))
            ]
        mis_energy_strain = [
            self.fitfun.mismatch_d_inner(
                data_strain['real'][i:].values + 1j*data_strain['imag'][i:].values,
                Re_strain[i] + 1j*Im_strain[i],
                data_strain['time'][i:].values,
            )
            for i in range(len(fit_strain))
            ]

        with open('data/overtone_fits/mismatch_strain.dat', 'w') as f:
            np.savetxt(
                f,
                np.column_stack((
                    data_strain['time'][:len(mis_strain)],
                    mis_strain,
                    )),
                delimiter='\t',
                header ='#(1)time (2)mismatch'
            )
        
        with open('data/overtone_fits/mismatch_energy_strain.dat', 'w') as f:
            np.savetxt(
                f,
                np.column_stack((
                    data_strain['time'][:len(mis_energy_strain)],
                    mis_energy_strain,
                    )),
                delimiter='\t',
                header = '#(1)time (2)mismatch'
                )

        mismatch = {
            'strain': {
                'time': data_strain['time'][:len(mis_strain)].values,
                'mismatch': np.array(mis_strain)
                },
            'energy_strain': {
                'time': data_strain['time'][:len(mis_energy_strain)].values,
                'mismatch': np.array(mis_energy_strain)
                },
            }
        return mismatch

    def compute_mismatch_dtheta(self, data_dtheta, fit_dtheta, omegas, modes):
        dtheta = [np.array(self.fitfun.d_theta(
                    data_dtheta['time'][i:] - data_dtheta['time'][i],
                    fit_dtheta[i].params['R'].value,
                    fit_dtheta[i].params['phi'].value,
                    omegas[modes[0]]['omega_r'],
                    omegas[modes[1]]['omega_r'],
                    omegas[modes[0]]['omega_i'],
                    omegas[modes[1]]['omega_i'],
                )) for i in range(len(fit_dtheta))]

        mis_dtheta = [
            self.fitfun.mismatch_inner(
                data_dtheta['dtheta'][i:].values,
                dtheta[i],
                data_dtheta['time'][i:].values,
            )
            for i in range(len(fit_dtheta))
            ]
        mis_energy_dtheta = [
            self.fitfun.mismatch_d_inner(
                data_dtheta['dtheta'][i:].values,
                dtheta[i],
                data_dtheta['time'][i:].values,
            )
            for i in range(len(fit_dtheta))
            ]

        with open('data/overtone_fits/mismatch_dtheta.dat', 'w') as f:
            np.savetxt(
                f,
                np.column_stack((
                    data_dtheta['time'][:len(mis_dtheta)],
                    mis_dtheta,
                    )),
                delimiter='\t',
                header ='#(1)time (2)mismatch'
            )

        with open('data/overtone_fits/mismatch_energy_dtheta.dat', 'w') as f:
            np.savetxt(
                f,
                np.column_stack((
                    data_dtheta['time'][:len(mis_energy_dtheta)],
                    mis_energy_dtheta,
                    )),
                delimiter='\t',
                header = '#(1)time (2)mismatch'
                )

        mismatch = {
            'dtheta': {
                'time': data_dtheta['time'][:len(mis_dtheta)].values,
                'mismatch': np.array(mis_dtheta)
                },
            'energy_dtheta': {
                'time': data_dtheta['time'][:len(mis_energy_dtheta)].values,
                'mismatch': np.array(mis_energy_dtheta)
                },
            }
        return mismatch

    def _compute_min_mismatch(
        self,
        mismatch:dict,
        ):
        mins = {}
        fit_pars = {}
        columns_fit = {
            'dtheta': ['time', 'R', 'dphi'],
            'strain': ['time', 'A_0', 'A_1', 'phi_0', 'phi_1', 'R', 'dphi']
            }

        for key, value in mismatch.items():

            aux = value['mismatch'][0]
            i = 1
            if key == 'energy_strain':
                i = int(mins['strain']['index']-20)
            while value['mismatch'][i] < aux:
                aux = value['mismatch'][i]
                i += 1
                    
            mins[key] = {
                'time': value['time'][value['time']==value['time'][i]],
                'mismatch': value['mismatch'][value['time']==value['time'][i]],
                'index': i,
            }
            try:
                fit_pars[key] = np.genfromtxt(f'data/overtone_fits/fitpar_{key}.dat')
                fit_pars[key] = pd.DataFrame(fit_pars[key], columns = columns_fit[key])

            except: pass
        return mins, fit_pars

    def find_best_fit(self, mins, fit_pars):
        best_pars = {
            'dtheta': fit_pars['dtheta'].loc[mins['energy_dtheta']['index']],
            'strain': fit_pars['strain'].loc[mins['strain']['index']],
        }
        
        for (key, value) in best_pars.items():
            with open(f'data/overtone_fits/best_fit_{key}.json', 'w') as fp:
                json.dump(value.to_dict(), fp, indent=4)
        return best_pars 

    def find_pars_time_dtheta(self, data_dtheta, best_pars, omegas):
        R_t = best_pars['dtheta']['R']*np.exp(
            (omegas['(2,2,1)']['omega_i'] - omegas['(2,2,0)']['omega_i'])*(
                best_pars['dtheta']['time'] - data_dtheta['time']
                )
            )
        dphi_t = (best_pars['dtheta']['dphi'] +
            (omegas['(2,2,0)']['omega_r']- omegas['(2,2,1)']['omega_r'])*(
                best_pars['dtheta']['time'] - data_dtheta['time']
            ))
        expected_pars = pd.DataFrame(np.array((
            data_dtheta['time'].values,
            R_t,
            dphi_t,
            )).T,
            columns=['time', 'R', 'dphi']
            )

        with open(f'data/overtone_fits/best_fit_dtheta_t10.json', 'w') as fp:
            json.dump({
                'time': 10,
                'R': best_pars['dtheta']['R']*np.exp(
                    (omegas['(2,2,1)']['omega_i'] - omegas['(2,2,0)']['omega_i'])*(
                        best_pars['dtheta']['time'] - 10
                    )),
                'dphi': (best_pars['dtheta']['dphi'] +
                (omegas['(2,2,0)']['omega_r']- omegas['(2,2,1)']['omega_r'])*(
                    best_pars['dtheta']['time'] - 10
                    ))
            }, fp, indent=4)

        return expected_pars

    def find_pars_time_strain(self, data_strain, best_pars, omegas):
        A_0 = best_pars['strain']['A_0']*np.exp(
            (omegas['(2,2,0)']['omega_i'])*(
                best_pars['strain']['time'] - data_strain['time']
                )
            )

        A_1 = best_pars['strain']['A_1']*np.exp(
            (omegas['(2,2,1)']['omega_i'])*(
                best_pars['strain']['time'] - data_strain['time']
                )
            )

        R_t = A_1/A_0

        phi_0 = (best_pars['strain']['phi_0'] +
            omegas['(2,2,0)']['omega_r']*(
                best_pars['strain']['time'] - data_strain['time']
            ))

        phi_1 = (best_pars['strain']['phi_1'] +
            omegas['(2,2,1)']['omega_r']*(
                best_pars['strain']['time'] - data_strain['time']
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

        A0_10 = best_pars['strain']['A_0']*np.exp(
            (omegas['(2,2,0)']['omega_i'])*(
                best_pars['strain']['time'] - 10
            ))

        A1_10 = best_pars['strain']['A_1']*np.exp(
                (omegas['(2,2,1)']['omega_i'])*(
                    best_pars['strain']['time'] - 10
                ))

        phi_0_10 = (best_pars['strain']['phi_0'] +
            omegas['(2,2,0)']['omega_r']*(
                best_pars['strain']['time'] - 10
            ))

        phi_1_10 = (best_pars['strain']['phi_1'] +
            omegas['(2,2,1)']['omega_r']*(
                best_pars['strain']['time'] - 10
            ))
        
        with open(f'data/overtone_fits/best_fit_strain_t10.json', 'w') as fp:
            json.dump({
                'time': 10,
                'R': A1_10/A0_10,
                'dphi': phi_0_10 - phi_1_10,
                'A_0': A0_10,
                'A_1': A1_10,
                'phi_0': phi_0_10,
                'phi_1': phi_1_10,
            }, fp, indent=4)

        return expected_pars


class FitFunctions:
    def __init__(self):
        pass
    # theta dot
    def d_theta(self, x, R, phi, wr0, wr1, wi0, wi1):
        return (exp(2*x*wi1)*wr0 + wr1*(R**2)*exp(2*x*wi0) + \
                R*exp(x*(wi0 + wi1))*((wr0+wr1)*cos(x*(wr0-wr1)-phi) + (-wi0+wi1)*sin(x*(wr0-wr1)-phi)))/(
                exp(2*x*wi1) + exp(2*x*wi0)*R**2 + 2*exp(x*(wi0+wi1))*R*cos(x*(wr0-wr1)-phi)
                )

    def residual_d_theta(self, params,  wr0=None, wr1=None, wi0=None, wi1=None, x=None, dat_d_theta=None):
        model_d_theta = self.d_theta(x, params['R'], params['phi'], wr0, wr1, wi0, wi1)
        resid_dtheta = dat_d_theta - model_d_theta
        return resid_dtheta

    # fundamental mode strain
    def f_Re_1_mode(self, x, A0, t0, wr0, wi0):
        return \
            A0*exp(-wi0*x)*cos(wr0*x-t0)

    def f_Im_1_mode(self, x, A0, t0, wr0, wi0):
        return \
            A0*exp(-wi0*x)*sin(wr0*x-t0)

    def residual_1_mode(self, params, wr0=None, wi0=None,  x=None, datRe=None, datIm=None):
        modelRe = self.f_Re_1_mode(x, params['A0r'], params['t0r'], wr0, wi0)
        modelIm = self.f_Im_1_mode(x, params['A0i'], params['t0i'], wr0, wi0)

        residRe = datRe - modelRe
        residIm = datIm - modelIm
        return np.concatenate((residRe, residIm))

    # fundamental mode + 1st overtone strain
    def f_Re_2_modes(self, x, A0, A1, t0, t1, wr0, wr1, wi0, wi1):
        return \
            A0*exp(-wi0*x)*cos(wr0*x-t0) + \
            A1*exp(-wi1*x)*cos(wr1*x-t1)

    def f_Im_2_modes(self, x, A0, A1, t0, t1, wr0, wr1, wi0, wi1):
        return \
            A0*exp(-wi0*x)*sin(wr0*x-t0) + \
            A1*exp(-wi1*x)*sin(wr1*x-t1)

    def residual_2_modes(self, params, wr0=None, wr1=None, wi0=None, wi1=None,  x=None, datRe=None, datIm=None):
        modelRe = self.f_Re_2_modes(x, params['A0r'], params['A1r'], params['t0r'], params['t1r'], wr0, wr1, wi0, wi1)
        modelIm = self.f_Im_2_modes(x, params['A0i'], params['A1i'], params['t0i'], params['t1i'], wr0, wr1, wi0, wi1)

        residRe = datRe - modelRe
        residIm = datIm - modelIm
        return np.concatenate((residRe, residIm))
    
    # mismatch
    def mismatch_inner(self, f, g, times):
        return 1 - np.abs(np.trapz(f*np.conj(g),times))/\
            np.sqrt(np.abs(np.trapz(f*np.conj(f),times))*np.abs(np.trapz(g*np.conj(g),times)))

    def mismatch_inner_dt(self, f, g, dt):
        return 1 - np.abs(sp.integrate.simps(f*np.conj(g),dt))/\
            np.sqrt(np.abs(sp.integrate.simps(f*np.conj(f),dt))*np.abs(sp.integrate.simps(g*np.conj(g),dt)))

    def mismatch_d_inner(self, f, g, times):
        return 1 - (np.abs(np.trapz((np.gradient(f, times))*(np.gradient(np.conj(g),times)), times)))/\
            np.sqrt(np.abs(np.trapz((np.gradient(f, times))*(np.gradient(np.conj(f), times)),times))*np.abs(
                np.trapz((np.gradient(g, times))*(np.gradient(np.conj(g), times)),times)))

    def mismatch_d_inner_dt(self, f, g, dt):
        return 1 - (np.abs(sp.integrate.simps((np.gradient(f, dt))*(np.gradient(np.conj(g),dt)),dt)))/\
            np.sqrt(np.abs(sp.integrate.simps((np.gradient(f, dt))*(np.gradient(np.conj(f), dt)),dt))*np.abs(
                sp.integrate.simps((np.gradient(g, dt))*(np.gradient(np.conj(g), dt)),dt)))


class Plots:
    def __init__(
        self,
        ):
        self.fitfun = FitFunctions()

    def plot_mismatch(self, mismatch, mins):
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
        plt.xlim(0, mismatch['dtheta']['time'][-100])
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
            x=mins['strain']['time'],
            color = colors['strain'],
            ls=':',
            lw=1
            )
        plt.axvline(
            x=mins['energy_dtheta']['time'],
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

    def plot_fitted_expected(self, fit_pars, best_pars, exp_pars, t_fin):
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

        ax1.axvline(best_pars['strain']['time'], color = 'red', ls=':', lw=1)
        ax1.axvline(best_pars['dtheta']['time'], color = 'black', ls=':', lw=1)

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

        ax3.axvline(best_pars['strain']['time'], color = 'red', ls=':', lw=1)
        ax3.axvline(best_pars['dtheta']['time'], color = 'black', ls=':', lw=1)

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

        ax2.axvline(best_pars['strain']['time'], color = 'red', ls=':', lw=1)
        ax2.axvline(best_pars['dtheta']['time'], color = 'black', ls=':', lw=1)

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

        ax4.axvline(best_pars['strain']['time'], color = 'red', ls=':', lw=1)
        ax4.axvline(best_pars['dtheta']['time'], color = 'black', ls=':', lw=1)

        plt.subplots_adjust(hspace=.001)
        fig.tight_layout()
        fig.savefig('figs/fit_pars.pdf', bbox_inches="tight")

    def plot_best_fits(self, best_pars, data_dtheta, data_strain, omegas, modes, t_initial):
        # plot best fit

        # compute best fits as a function of time
        strain_n1_fitted = abs(
            self.fitfun.f_Re_2_modes(
                data_strain['time'] - best_pars['strain']['time'],
                best_pars['strain']['A_0'],
                best_pars['strain']['A_1'],
                best_pars['strain']['phi_0'],
                best_pars['strain']['phi_1'],
                omegas[modes[0]]['omega_r'],
                omegas[modes[1]]['omega_r'],
                omegas[modes[0]]['omega_i'],
                omegas[modes[1]]['omega_i'],
                )
            + 1j*self.fitfun.f_Im_2_modes(
                data_strain['time'] - best_pars['strain']['time'],
                best_pars['strain']['A_0'],
                best_pars['strain']['A_1'],
                best_pars['strain']['phi_0'],
                best_pars['strain']['phi_1'],
                omegas[modes[0]]['omega_r'],
                omegas[modes[1]]['omega_r'],
                omegas[modes[0]]['omega_i'],
                omegas[modes[1]]['omega_i'],
                )
            )

        dtheta_fitted = self.fitfun.d_theta(
            data_dtheta['time'] - best_pars['dtheta']['time'],
            best_pars['dtheta']['R'],
            best_pars['dtheta']['dphi'],
            omegas[modes[0]]['omega_r'],
            omegas[modes[1]]['omega_r'],
            omegas[modes[0]]['omega_i'],
            omegas[modes[1]]['omega_i'],
            )

        strain_n0_fitted = abs(
            self.fitfun.f_Re_1_mode(
                data_strain['time'] - best_pars['strain']['time'],
                best_pars['strain']['A_0'],
                best_pars['strain']['phi_0'],
                omegas[modes[0]]['omega_r'],
                omegas[modes[0]]['omega_i'],
                )
            + 1j*self.fitfun.f_Im_1_mode(
                data_strain['time'] - best_pars['strain']['time'],
                best_pars['strain']['A_0'],
                best_pars['strain']['phi_0'],
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
        ax1.axvline(best_pars['strain']['time'], color = 'red', ls=':', lw=1)
        

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
        ax2.axvline(best_pars['strain']['time'], color = 'red', ls=':', lw=1)

        # dtheta fits
        ax3.set_ylabel(r'$\dot{\theta}_{22} [1/M]$', fontsize=font_size)
        ax3.set_xlim([0, 60])
        ax3.plot(data_dtheta['time'], data_dtheta['dtheta'], color='deepskyblue', lw=lw, label=r'NR')
        ax3.axhline(y=omegas[modes[0]]['omega_r'], color='limegreen', linewidth=lw, ls='--', label=r'$n = 0$')
        ax3.plot(data_dtheta['time'], dtheta_fitted, ls='--', color='k', lw=lw, label=r'$n = 0 + 1$')
        ax3.axvline(best_pars['dtheta']['time'], color = 'black', ls=':', lw=1)
        ax3.axvline(t_initial, ls=':', color='limegreen')
        ax3.legend(bbox_to_anchor=(1, 0.02), loc='lower right', fontsize=font_size, fancybox=True, framealpha=1)

        # dtheta relative differences
        ax4.set_ylim(bottom=pow(10, -5), top=pow(10, 0))
        ax4.set_xlabel(r'$t - t_{\mathrm{peak}} [M]$', fontsize=font_size)
        ax4.set_xlim([0, 60])
        ax4.set_yscale('log')
        ax4.plot(data_dtheta['time'], abs(data_dtheta['dtheta'] - omegas[modes[0]]['omega_r'])/data_dtheta['dtheta'], color='limegreen', lw=lw, ls='--')
        ax4.plot(data_dtheta['time'], abs(data_dtheta['dtheta'] - dtheta_fitted)/data_dtheta['dtheta'], color='k', lw=lw, ls='--')
        ax4.axvline(best_pars['dtheta']['time'], color = 'black', ls=':', lw=1)
        ax4.axvline(t_initial, ls=':', color='limegreen')

        plt.subplots_adjust(hspace=.001)
        fig.tight_layout()
        fig.savefig('figs/best_fits.pdf', bbox_inches="tight")

