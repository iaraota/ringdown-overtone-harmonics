import os
import json
import pandas as pd
import numpy as np

class ComputeOmegas:

    def __init__(
        self,
        ):
        pass

    def compute_omega(
        self,
        spin:float,
        lmn_modes:list,
        label:str,
        ):
        with open('import_data/metadata.json') as f:
            par_simu = json.load(f)

        a_M = round(spin,4)
        omegas = {}
        omegas_Mi = {}

        for (l,m,n) in lmn_modes:
            df = self.create_a_over_M_omegas_dataframe(l,m,n)
            mode = f'({l},{m},{n})'
            omega_r, omega_i = self.transform_mass_spin_to_omegas(a_M, df)
            omegas[mode] = {
                'omega_r': omega_r,
                'omega_i': omega_i,
            }
            omegas_Mi[mode] = {
                'omega_r': omega_r/par_simu['remnant_mass'],
                'omega_i': omega_i/par_simu['remnant_mass'],
            }

        # create folder data/omegas
        if not os.path.exists('data/omegas'):
            os.makedirs('data/omegas')

        with open(f'data/omegas/Mf_qnm_freqs_{label}.json', 'w') as fp:
            json.dump(omegas, fp, sort_keys=True, indent=4)

        with open(f'data/omegas/Mi_qnm_freqs_{label}.json', 'w') as fp:
            json.dump(omegas_Mi, fp, sort_keys=True, indent=4)

        return omegas_Mi

    def create_a_over_M_omegas_dataframe(
        self,
        l:float,
        m:float,
        n:float,
        ):
        if m<0:
            m = f'm{abs(m)}'
        files = np.genfromtxt(f'import_data/frequencies_l{l}/n{n+1}l{l}m{m}.dat', usecols=range(3))

        df = pd.DataFrame({"omega_r": files[:,1], "omega_i": -files[:,2]}, index = files[:,0])

        return df


    def transform_mass_spin_to_omegas(
        self,
        a_over_M:float,
        df,
        ):
        """Transform mass and spin do quasinormal mode omegas (frequencies)

        Parameters
        ----------
        M : float
            Black hole final mass in units of final mass.
            (M_final/M_initial)
        a : float
            Black hole spin in units of final mass.
        fit_coeff : array_like
            Fits coefficient to Kerr QNM frequencies. 
            See transf_fit_coeff method or  
            https://pages.jh.edu/eberti2/ringdown/

        Returns
        -------
        float, float
            Quasinormal mode frequencies in units of final mass.
        """
        omega_r = df.loc[round(a_over_M,4)].omega_r
        omega_i = df.loc[round(a_over_M,4)].omega_i

        return omega_r, omega_i

