import numpy as np

params = {
    "simulation_id": "k004_SUPER",
    "system_params": {
        "num_atoms": 900,
        "kappa": 0.55,
        "b": [0, 1e-3, 1e-2, 1e-1, 1],
        "t_final": 10000,
        "pbc": False
    },
    "simulation_params": {
        "runs": 9999,
        "max_step_number": False,
        "dt_save": 0.1,
        "ELWE_partition": 'epyc-256',  # epyc-256, physik-fleischhauer, skylake-96
        "memory_per_cpu": '10G',
        "use_project_TARDIS": True
    },
    "initial_state": "all_excited",  # single_center_seed, all_excited, all_ground
    "observables": ["state_dynamics", "avalanche_exponents", "runtime_seconds"]
    # Supported observables: state_dynamics, avalanche_exponents, runtime_seconds
}
