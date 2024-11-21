import os
import sys
import math
import numpy as np
import time as time_module
from datetime import datetime as dt


def visualize():
    pass

    # Visualization if desired
    # index_zr = states == 2
    # if np.any(index_dn):
    #     plt.scatter(self.pos[index_dn, 0], self.pos[index_dn, 1], color='blue', s=500)
    # if np.any(index_up):
    #     plt.scatter(self.pos[index_up, 0], self.pos[index_up, 1], color='red', s=500)
    # if np.any(index_zr):
    #     plt.scatter(self.pos[index_zr, 0], self.pos[index_zr, 1], color='black', s=500)
    #
    # plt.title(time)
    # plt.pause(1.)
    # plt.close()


class KineticMonteCarlo:
    
    def __init__(self, run_id: int, sim_id: str = None, job_id: int = None):

        self.run_id = run_id

        self.sim_id = sim_id
        self.job_id = job_id

        print(f'{dt.now().strftime("%Y-%m-%d %H:%M:%S")} - '
              f'Starting simulation {self.sim_id}_id{self.job_id}_run{self.run_id}.', flush=True)

        if job_id is None:
            self.params = np.load(os.path.join('k_data', f'{sim_id}_params.npy'), allow_pickle=True)[()]
        else:
            self.params = {}
            varied_params = np.load(os.path.join('k_data', f'{sim_id}_params.npy'), allow_pickle=True)[()]
            top_level_keys = ["system_params", "simulation_params"]
            for top_level_key in top_level_keys:

                dict_at_top_level_key = {}
                keys = list(varied_params[top_level_key].keys())

                for key in keys:

                    value = varied_params[top_level_key][key]
                    if type(value) is list or type(value) is np.ndarray:
                        static_value = value[job_id]
                        dict_at_top_level_key.update(
                            {key: static_value}
                        )
                    else:
                        dict_at_top_level_key.update(
                            {key: value}
                        )

                self.params.update({
                    top_level_key: dict_at_top_level_key
                })

            self.observables = varied_params['observables']
            self.params.update({
                "simulation_id": sim_id,
                "initial_state": varied_params['initial_state'],
                "observables": self.observables
            })

        # Unpack system parameters
        system_params = self.params['system_params']
        self.num_atoms = system_params['num_atoms']
        self.kappa = system_params['kappa']
        self.b = system_params['b']
        self.t_final = system_params['t_final']
        self.pbc = system_params['pbc']

        # Unpack simulation parameters
        simulation_params = self.params['simulation_params']
        self.max_step_number = simulation_params['max_step_number']
        self.rng = np.random.default_rng(run_id)
        self.dt_save = simulation_params['dt_save']

        # Unpack initial condition
        self.init_cond = self.params['initial_state']

        # Initialize class properties
        self.lattice_length = None
        self.pos = None
        self.dist = None

    def generate_two_dim_lattice(self):
        # Get lattice length using integer square root function from math module
        self.lattice_length = math.isqrt(self.num_atoms)

        # Check if num atoms is a perfect square
        if self.num_atoms != self.lattice_length ** 2:
            raise ValueError(f'Error, self.num_atoms must be a perfect square! '
                             f'However, self.num_atoms = {self.num_atoms} is not.')

        # Create positions vector
        atom_index = np.arange(self.num_atoms)
        self.pos = np.empty(shape=(self.num_atoms, 2), dtype=float)
        self.pos[:, 0] = atom_index % self.lattice_length
        self.pos[:, 1] = np.floor(atom_index / self.lattice_length)

        # Create distance matrix
        self.dist = np.abs(self.pos[:] - self.pos[:, np.newaxis])
        if self.pbc:
            self.dist[:] = np.where(self.dist > self.lattice_length / 2, self.lattice_length - self.dist, self.dist)
        self.dist *= self.dist
        self.dist = np.sqrt(np.sum(self.dist, axis=-1))

    def _set_initial_conditions(self, states, excitation_tracker):

        if self.init_cond == 'single_center_seed':

            # Excite central atom
            x_center = (self.lattice_length - 1) / 2
            y_center = (self.lattice_length - 1) / 2
            dist_to_center = np.linalg.norm(self.pos - [x_center, y_center], axis=1)
            index_center = np.argmin(dist_to_center)
            states[index_center] = 1

            excitation_tracker[index_center] = 1

        elif self.init_cond == 'all_ground':
            states[:] = 0

        elif self.init_cond == 'all_excited':
            states[:] = 1

            excitation_tracker[:] = 1

        else:
            raise ValueError(f'Initial condition {self.init_cond} not supported. Maybe check spelling?')

        return states, excitation_tracker

    def simulate(self):

        # NO stimulated decay
        # NO off-resonant excitations
        # ONLY NN interactions. No "diagonal blockade"

        t0 = time_module.time()

        step_number = 0
        time = 0

        # Initialize states and rates
        states = np.zeros(shape=self.num_atoms, dtype=int)
        rates = np.zeros(shape=2 * self.num_atoms, dtype=float)

        # Initiate save_arrays
        save_times = np.arange(0, self.t_final, self.dt_save)
        save_states = np.full(shape=(len(states), len(save_times)), fill_value=np.nan)
        save_index = 0

        # Initialize observables for avalanche exponents: (i) time, (ii) area (UNIQUE), (iii) size (TOTAL)
        t_fin = np.nan
        num_times_atom_was_excited = np.zeros(shape=self.num_atoms, dtype=int)

        # Initial conditions
        states, num_times_atom_was_excited = self._set_initial_conditions(states, num_times_atom_was_excited)

        # Add initial conditions to save arrays
        save_times[0] = 0
        save_states[:, 0] = states
        save_index += 1

        while time < self.t_final:

            rates *= 0

            # Determine rates
            index_up = states == 1
            id_up = np.argwhere(index_up).flatten()
            index_dn = states == 0

            if np.sum(index_up) == 0:
                print(f'No more Rydberg atoms at time t={time}.')
                if "avalanche_exponents" in self.observables:
                    t_fin = time

                for i in range(save_index, len(save_times)):
                    save_states[:, i] = states
                break

            # Calculate decay rates
            rates[2 * id_up] = 1 - self.b  # A spin-up atom decays at rate 1-b to the ground state
            rates[2 * id_up + 1] = self.b  # And it decays at rate b to the removed state

            # Calculate excitation rates
            if np.any(index_up):
                in_r_fac = self.dist[index_up, :] == 1  # Get number of excited NN (this line and the next)
                num_ryd_in_r_fac = np.sum(in_r_fac, axis=0)
                does_it_fac = index_dn & (num_ryd_in_r_fac == 1)  # Atom can be excited if: down state AND 1 excited NN
                id_does_it_fac = np.argwhere(does_it_fac).flatten()

                rates[2 * id_does_it_fac] = self.kappa

            cum_sum = np.cumsum(rates)
            total_rate = cum_sum[-1]
            rng = self.rng.random() * total_rate

            transition_index = np.argwhere(cum_sum > rng)[0][0]

            if transition_index % 2 == 0:
                # Even transition index means:
                # A ground state atom is being excited
                # OR a Rydberg state atom was selected to decay to the ground state
                atom_id = transition_index // 2
                if states[atom_id] == 0:
                    states[atom_id] = 1
                    num_times_atom_was_excited[atom_id] += 1
                elif states[atom_id] == 1:
                    states[atom_id] = 0
                else:
                    raise ValueError('The code malfunctioned and selected a state 2 atom to transition.')
            else:
                # Odd transition index means:
                # A ground state should not be able to be selected
                # A Rydberg state atom was selected to decay to the removed state
                atom_id = (transition_index - 1) // 2
                if states[atom_id] == 0:
                    raise ValueError('The code malfunctioned. Selected state 0 with odd transition_index.')
                elif states[atom_id] == 1:
                    states[atom_id] = 2
                else:
                    raise ValueError('The code malfunctioned and selected a state 2 atom to transition.')

            # Increase time
            u = self.rng.uniform()
            tau = -np.log(u) / total_rate

            time += tau

            # Save states
            if time > save_times[save_index]:

                new_save_index = np.max(np.argwhere(save_times < time).flatten())

                if new_save_index > save_index:
                    for i in range(save_index, new_save_index):
                        save_states[:, i] = save_states[:, save_index - 1]

                save_states[:, new_save_index] = states

                save_index = new_save_index + 1

                if save_index >= len(save_times):
                    break

            if type(self.max_step_number) is int:
                if step_number > self.max_step_number:
                    print(f'Warning: max steps of {step_number} exceeded!', flush=True)
                    for i in range(save_index, len(save_times)):
                        save_states[:, i] = states
                    break

        if time < save_times[-1]:
            for i in range(save_index, len(save_times)):
                save_states[:, i] = save_states[:, save_index]

        if self.job_id is None:
            f_path = os.path.join('k_data_raw', f'{self.sim_id}_raw',
                                  f'{self.sim_id}_run{self.run_id}.npz')
        else:
            f_path = os.path.join('k_data_raw', f'{self.sim_id}_raw',
                                  f'{self.sim_id}_id{self.job_id}_run{self.run_id}.npz')

        t1 = time_module.time()

        print(f'{dt.now().strftime("%Y-%m-%d %H:%M:%S")} - '
              f'Finished simulation {self.sim_id}_id{self.job_id}_run{self.run_id} in {round(t1-t0, 3)} seconds.',
              flush=True)

        data = {}

        if "avalanche_exponents" in self.observables:

            unique_ryds = np.sum(num_times_atom_was_excited > 0)
            total_ryds = np.sum(num_times_atom_was_excited)

            data.update({
                "avalanche_time": t_fin,
                "avalanche_area": unique_ryds,
                "avalanche_size": total_ryds
            })

        if "state_dynamics" in self.observables:

            num_grd = np.sum(save_states == 0, axis=0)
            num_ryd = np.sum(save_states == 1, axis=0)
            num_ded = np.sum(save_states == 2, axis=0)

            data.update({
                "time": save_times,
                "num_grd": num_grd,
                "num_ryd": num_ryd,
                "num_ded": num_ded
            })

        if "runtime_seconds" in self.observables:

            runtime = t1 - t0

            data.update({
                "runtime_seconds": runtime
            })

        np.savez(f_path, data=data, params=self.params)
