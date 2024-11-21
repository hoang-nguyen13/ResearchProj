import os
import itertools
import numpy as np
from monte_carlo_simulation import KineticMonteCarlo


def parse_params():
    import parameters

    # Unpack parameters. Only check in system_params and simulation_params for varied values
    params = parameters.params
    relevant_params = params["system_params"].copy()
    relevant_params.update(params["simulation_params"])
    relevant_params.update({"initial_state": params["initial_state"]})

    # Initiate lists for the names and values of parameters which are iterated, i.e. they are lists or arrays
    iterated_param_keys = []
    iterated_param_values = []

    # Iterate over all relevant parameters to find the iterated parameters
    for key in relevant_params:
        value = relevant_params[key]

        if key == 'heap_size_hint':
            pass
        else:
            if value is None:
                raise ValueError(f'Key "{key}" is None. This is unsupported. Check parameters.py')
            elif isinstance(value, (np.ndarray, list)):
                iterated_param_keys.append(key)
                iterated_param_values.append(value)
                print(f'Detected ITERATED parameter with name "{key}".', flush=True)
            elif isinstance(value, (int, float, str)):
                print(f'Detected constant parameter "{key}" with value: {value}.', flush=True)
            else:
                raise ValueError(f'Key "{key}" is unsupported instance with type {type(key)}. Check kmc_runner.py')

    # If iterated parameters exist, create a meshgrid of them with each tuple receiving a unique id
    if len(iterated_param_keys) == 0:
        cartesian_product = None
        ids_ = None
        params_with_product = params.copy()
    else:
        cartesian_product = list(itertools.product(*iterated_param_values))
        ids_ = np.arange(len(cartesian_product))

        # Return a new parameters dictionary with the cartesian product
        params_with_product = params.copy()
        for i, key in enumerate(iterated_param_keys):
            if key in params["simulation_params"]:
                top_level_key = "simulation_params"
            elif key in params["system_params"]:
                top_level_key = "system_params"
            elif key in params["initial_state"]:
                top_level_key = "initial_state"
            else:
                raise ValueError(f'Iterated key "{key}" not found in simulation_params or system_params.')

            cartesian_list = [l_[i] for l_ in cartesian_product]
            params_with_product[top_level_key][key] = cartesian_list

            assert set(cartesian_list) == set(iterated_param_values[i]), "Cartesian list and iterated param list do" \
                                                                         "not contain the same values!"

    params_with_product['job_ids'] = ids_

    return ids_, cartesian_product, params_with_product


def setup(job_ids, params_dict: dict):
    import subprocess

    sim_id = params_dict["simulation_id"]
    runs = params_dict['simulation_params']['runs']

    # Check if the code is being executed locally or on the ELWE cluster
    if 'scratch' in os.listdir('/'):
        print('Runner seems to be running on ELWE.', flush=True)

        # Save parameters into the new path
        print('KMC Runner - Saving parameters.', flush=True)
        path_dict = os.path.join('/', 'home', 'brady', 'rydberg_lattice', 'k_data', f'{sim_id}_params')
        np.save(path_dict, params_dict)

        # Create raw directories
        raw_path = os.path.join('/', 'home', 'brady', 'rydberg_lattice', 'k_data_raw', f'{sim_id}_raw')
        if os.path.exists(raw_path):
            pass
        else:
            os.mkdir(raw_path)

        # Edit SLURM job script to incorporate a slurm array of length ids and have the name "sim_id"
        # Create SLURM job script to incorporate all parameters
        print('KMC Runner - Editing SLURM script "auto_slurm_runner".', flush=True)

        partition = params_dict["simulation_params"]["ELWE_partition"]
        print(f'Partition "{partition}" has been specified.', flush=True)
        mem_per_cpu = params_dict["simulation_params"]["memory_per_cpu"]
        use_project_tardis = params_dict["simulation_params"]["use_project_TARDIS"]

        slurm_script_lines = [
            '#!/bin/bash \n',
            '\n',
            '\n',
            f'#SBATCH -J {sim_id}\n',
            '#SBATCH -o logs_auto_runner/%x_log.out \n',
            '#SBATCH -e logs_auto_runner/%x_log.err \n',
            '#SBATCH --cpus-per-task=1 \n',
            '#SBATCH --time=200:00:00 \n',
            f'#SBATCH --mem-per-cpu={mem_per_cpu} \n',
            f'#SBATCH -p {partition}\n',
            '#SBATCH --mail-type=FAIL,END \n',
            '\n',
            '\n',
            '\n',
            '\n'
        ]

        if use_project_tardis:
            slurm_script_lines[2] = '#SBATCH --account=RPTU-TARDIS \n'

        if job_ids is None:
            slurm_script_lines[11] = f'#SBATCH --array=0-{runs}\n'
            slurm_script_lines[12] = 'id=${SLURM_ARRAY_TASK_ID} \n'
            slurm_script_lines[13] = 'module load anaconda3/latest\n'
            slurm_script_lines[14] = 'python -m main ${SLURM_JOB_NAME} ${id} 0 0'
        else:
            num_jobs = int(len(job_ids) * runs)
            slurm_script_lines[11] = f'#SBATCH --array=0-{num_jobs}\n'
            slurm_script_lines[12] = 'id=${SLURM_ARRAY_TASK_ID} \n'
            slurm_script_lines[13] = 'module load anaconda3/latest \n'
            slurm_script_lines[14] = 'python -m main ${SLURM_JOB_NAME} ${id}' + f' {len(job_ids)} {runs}'

        # Save the modified SLURM script
        print('KMC Runner- Saving modified SLURM script.', flush=True)
        with open('auto_slurm_runner', 'w') as f:
            f.writelines(slurm_script_lines)

        # Run SLURM job script
        print('KMC Runner - Launching SLURM script.', flush=True)
        subprocess.run(['sbatch', 'auto_slurm_runner'])

    else:
        print('Runner seems to be running locally.', flush=True)

        # Save parameters into the new path
        print('KMC Runner - Saving parameters.', flush=True)
        path_dict = os.path.join('k_data', f'{sim_id}_params.npy')
        np.save(path_dict, params_dict)

        # Create raw directories
        raw_path = os.path.join('k_data_raw', f'{sim_id}_raw')
        if os.path.exists(raw_path):
            pass
        else:
            os.mkdir(raw_path)

        # Run Python main script
        print('KMC Runner - Launching Python script.', flush=True)
        if job_ids is None:
            for run_id in range(params_dict['simulation_params']['runs']):
                f = KineticMonteCarlo(run_id=run_id, sim_id=sim_id)
                f.generate_two_dim_lattice()
                f.simulate()

        else:
            raise ValueError('Local running currently does not support varying parameters!')


if __name__ == '__main__':

    print(' -- Welcome to KMC Runner! -- ', flush=True)

    print('KMC Runner - Parsing parameters.', flush=True)
    ids, param_values_product, params_dict_product = parse_params()

    print('KMC Runner - Creating save directory and preparing SLURM scripts.', flush=True)
    setup(job_ids=ids, params_dict=params_dict_product)
