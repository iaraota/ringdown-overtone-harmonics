import json
import os 

import numpy as np
import h5py # hdf5 library
import pandas as pd

class ImportSimulation():
    """Import simulation data and save to files.
    """
    def __init__(
        self,
        file_path:str,
        lm_modes:list,
        lmn_modes:list,
        lm_dominant:list,
        ):
        """Import SXS simulation waveform and save
        to files. Import the simulation parameters,
        compute the QNM frequencies and save to file.

        Parameters
        ----------
        file_path : str
            path containing simulation data.
        lm_modes : list
            (l, m) modes to import from simulation.
        lmn_modes : list
            (l,m,n) modes to compute QNM frequencies.
        """
        self.save_par_simu_to_files(file_path, lm_modes,lm_dominant)
        # self.compute_simulation_QNM_frequencies(file_path, lmn_modes)

    def compute_d_theta(
        self,
        data:dict,
        label_lm:str,
        label_data:str,
        ):
        """Compute the derivative of the complex phase of
        the waveform: z = |z|e^{i*theta}, where theta is
        the complex phase. If one QNM describes the whole
        waveform, the derivative of theta equals the real
        frequency of the QNM. If the waveform is a sum of
        overtones, after some time all the overtones will
        be neglegible due to fast decay and the derivative
        of theta will be the fundamental mode real frequency.


        Parameters
        ----------
        data : dictionary
            Waveform data, keys: 'time' as the time array,
            'real' as the real polarization of the waveform
            and 'imag' as the imaginary polarization.
        label_lm : string
            label for the harmonic (l,m) for file name.
        label_data : string
            extra label for file name, can indicate which
            part of the waveform is used (eg., peak).
        """
        ### z = |z|e^{i*phi}

        phi = np.arctan(-data['imag'] / data['real'])
        # make tetha continuous

        phi_diver = []
        for i in range(len(phi) - 1):
            if phi[i + 1] < phi[i]:
                phi_diver.append(i)
        n = 0
        for i in range(len(phi_diver)):
            for j in range(len(phi)):
                if j - 1 == phi_diver[i]:
                    n = n + np.pi
                phi[j] = phi[j] + n
        del n, phi_diver
        # save to file derivative of fit_overtone
        d_arctan = np.column_stack((data['time'][0:-1], np.diff(phi) / np.diff(data['time'])))
        
        # create folder data/waveforms
        if not os.path.exists('data/d_theta'):
            os.makedirs('data/d_theta')

        open(f'data/d_theta/{label_data}_{label_lm}.dat', 'w').close()  # clear file
        np.savetxt(
            f'data/d_theta/{label_data}_{label_lm}.dat',
            d_arctan,
            delimiter='\t',
            header='# (1) time (2) derivative of fit_overtone',
            )

    def save_par_simu_to_files(
        self,
        file_path:str,
        lm_modes:list,
        lm_dominant:list=(2,2)
        ):
        """Import waveform from SXS simulation and save to files.
        Also compute the derivative of the phase and save to file,
        by calling the method compute_d_theta.
        File should be 'rhOverM_Asymptotic_GeometricUnits_CoM.h5' 
        from SXS catalog.

        Parameters
        ----------
        file_path : str
            Path containing the simulation files.
        lm_modes : list
            List containing the desired imported modes. Should take
            the form: ((l1,m2), (l2,m2), ...)
            For single mode use list: [(l,m)]

        lm_dominant : list
            (l,m) dominant mode used do choose the amplitude peak. 
            Should be a list such that lm[0] = l and lm[1] = m.
            By default: (2,2)

        """
        ####################################################################################################################
        # import data
        # assumes it is a SXS rhOverM_Asymptotic_GeometricUnits_CoM.h5 file
        hdf5_file = h5py.File(file_path+'/rhOverM_Asymptotic_GeometricUnits_CoM.h5', 'r')  # import hdf5 file
        files_dic = {}
        for (l, m) in lm_modes:
            # insert data file to an array (1)time(2)real-part(3)imaginary-part
            files_dic[f'l{l}m{m}'] = np.asarray(list(hdf5_file[f'OutermostExtraction.dir/Y_l{l}_m{m}.dat']))

        # create folder data/waveforms
        if not os.path.exists('data/waveforms'):
            os.makedirs('data/waveforms')

        # save to file
        header = '# (1) time \t (2) real part \t(3) imaginary part'
        for key, value in files_dic.items():
            np.savetxt('data/waveforms/all_{}_sim.dat'.format(key), value, delimiter='\t', header = header)

        ####################################################################################################################
        # remove everything before the peak
        # find the peak relative to the amplitude of the wave
        final_time_index = 1000
        posi_peak, peak_strain, peak_dominant_strain = {}, {}, {}

        key_dominant = f'l{lm_dominant[0]}m{lm_dominant[1]}'
        posi_peak[key_dominant] = np.concatenate(
            np.where(
                np.sqrt(
                    files_dic[key_dominant][:, 1] ** 2 + files_dic[key_dominant][:, 2] ** 2) ==
                    np.max(np.sqrt(files_dic[key_dominant][:, 1] ** 2 + files_dic[key_dominant][:, 2] ** 2)
                    )
                )
            )[0]
        for key, value in  files_dic.items():
            # find peak of amplitude
            posi_peak[key] = np.concatenate(
                np.where(
                    np.sqrt(
                        value[:, 1] ** 2 + value[:, 2] ** 2) ==
                        np.max(np.sqrt(value[:, 1] ** 2 + value[:, 2] ** 2)
                        )
                    )
                )[0]

            # remove points before peak and set t_peak = 0
            ## before peak of the harmonic key
            peak_strain[key] = np.column_stack((
                value[posi_peak[key]:, 0] - value[posi_peak[key]][0],
                value[posi_peak[key]:, 1], value[posi_peak[key]:, 2]
                ))[:final_time_index]

            ## before peak of the dominant harmonic (l,m) 
            peak_dominant_strain[key] = np.column_stack((
                value[posi_peak[key_dominant]:, 0] - value[posi_peak[key_dominant]][0],
                value[posi_peak[key_dominant]:, 1], value[posi_peak[key_dominant]:, 2]
                ))[:final_time_index]

            # save to file
            np.savetxt(f'data/waveforms/peak_{key}.dat', peak_strain[key], delimiter='\t', header=header)
            np.savetxt(f'data/waveforms/peak_{key_dominant}_{key}.dat', peak_dominant_strain[key], delimiter='\t', header=header)

            peak_strain[key] = {'time': peak_strain[key][:,0], 'real': peak_strain[key][:,1], 'imag': peak_strain[key][:,2]}
            peak_dominant_strain[key] = {'time': peak_dominant_strain[key][:,0], 'real': peak_dominant_strain[key][:,1], 'imag': peak_dominant_strain[key][:,2]}

            self.compute_d_theta(peak_strain[key], key, 'peak')
            self.compute_d_theta(peak_dominant_strain[key], key, f'peak_{key_dominant}')
