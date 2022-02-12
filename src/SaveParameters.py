import json
import os
import numpy as np

class SaveParameters:

    def __init__(
        self,
        file_path,
        lm_modes,
        lm_dominant
        ):

        # create folder data/qnm_pars
        if not os.path.exists('data/qnm_pars'):
            os.makedirs('data/qnm_pars')

        harm_fits = self.import_harmonics(lm_modes)
        harm_fits = self.compute_ratios_dphis(harm_fits, lm_dominant)

        over_fits = self.import_overtone_fit(lm_dominant)
        over_fits = self.compute_A_phase_dtheta(harm_fits, over_fits, lm_dominant)

        self.import_simulation(file_path)

        self.save_pars(harm_fits,over_fits)

        # import simulation frequencies
        with open('data/omegas/Mi_qnm_freqs_sim.json') as file:
            omegas = json.loads(file.read())    

        with open(f'data/qnm_pars/omegas.json', 'w') as fp:
            json.dump(omegas, fp, sort_keys=True, indent=4)

    def save_pars(
        self,
        harm_fits,
        over_fits,
        ):
        ratios = {}
        dphis = {}
        amplitudes = {}
        phases = {}
        for key in harm_fits:
            ratios[key] = harm_fits[key]['R']
            dphis[key] = harm_fits[key]['dphi']
            phases[key] = harm_fits[key]['phi']
            amplitudes[key] = abs(harm_fits[key]['A'])

        for key in over_fits:
            ratios[key] = over_fits[key]['R']
            dphis[key] = over_fits[key]['dphi']
            amplitudes[key] = abs(over_fits[key]['A_1'])
            phases[key] = over_fits[key]['phi_1']

        with open(f'data/qnm_pars/ratios.json', 'w') as fp:
            json.dump(ratios, fp, sort_keys=True, indent=4)

        with open(f'data/qnm_pars/dphi.json', 'w') as fp:
            json.dump(dphis, fp, sort_keys=True, indent=4)

        with open(f'data/qnm_pars/amplitudes.json', 'w') as fp:
            json.dump(amplitudes, fp, sort_keys=True, indent=4)

        with open(f'data/qnm_pars/phases.json', 'w') as fp:
            json.dump(phases, fp, sort_keys=True, indent=4)


    def compute_ratios_dphis(
        self,
        harm_fits,
        lm_dominant,
        ):
        (l, m) = lm_dominant
        for key in harm_fits:
            harm_fits[key]['R'] = abs(harm_fits[key]['A']/
                harm_fits[f'({l},{m},0)']['A'])
            harm_fits[key]['dphi'] = harm_fits[f'({l},{m},0)']['phi'] - harm_fits[key]['phi']

            while harm_fits[key]['dphi'] < 0:
                harm_fits[key]['dphi'] += 2*np.pi
            while harm_fits[key]['dphi'] > 2*np.pi:
                harm_fits[key]['dphi'] -= 2*np.pi

        return harm_fits

    def compute_A_phase_dtheta(
        self,
        harm_fits,
        over_fits,
        lm_dominant,
        ):
        (l, m) = lm_dominant
        over_fits[f'({l},{m},1) I']['A_1'] = abs(over_fits[f'({l},{m},1) I']['R']*
                harm_fits[f'({l},{m},0)']['A'])
        over_fits[f'({l},{m},1) I']['phi_1'] = (harm_fits[f'({l},{m},0)']['phi'] 
                - over_fits[f'({l},{m},1) I']['dphi'])
        return over_fits

    def import_harmonics(
        self,
        lm_modes,
        ):
        harm_fits = {}
        for (l,m) in lm_modes:
            with open(f'data/fits/best_fit_t10_l{l}m{m}.json') as f:
                harm_fits[f'({l},{m},0)'] = json.load(f)
        
        for key in harm_fits:
            while harm_fits[key]['phi'] > 2*np.pi:
                harm_fits[key]['phi'] -= 2*np.pi
            
            while harm_fits[key]['phi'] < 0:
                harm_fits[key]['phi'] += 2*np.pi

        return harm_fits    

    def import_overtone_fit(
        self,
        lm_dominant,
        ):
        (l, m) = lm_dominant
        over_fits = {}
        with open(f'data/overtone_fits/best_fit_dtheta_t10.json') as f:
            over_fits[f'({l},{m},1) I'] = json.load(f)

        with open(f'data/overtone_fits/best_fit_strain_t10.json') as f:
            over_fits[f'({l},{m},1) II'] = json.load(f)
        
        return over_fits

    def import_simulation(
        self,
        file_path,
        ):
        with open(file_path+'/metadata.json') as f:
            par_simu = json.load(f)

        par_simu['final_spin'] = np.sqrt(sum(x**2 for x in par_simu['remnant_dimensionless_spin']))


        with open(f'data/qnm_pars/bh_pars.json', 'w') as fp:
            json.dump({
                'remnant_mass': par_simu['remnant_mass'],
                'remnant_spin': par_simu['final_spin'],
                },
                fp, sort_keys=True, indent=4)
