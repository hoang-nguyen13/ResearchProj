import sys
import numpy as np
from monte_carlo_simulation import KineticMonteCarlo


sim_id = sys.argv[1]
slurm_id = int(sys.argv[2])
num_job_ids = int(sys.argv[3])
num_runs = int(sys.argv[4])

if num_job_ids == 0:
    run_id = slurm_id
    job_id = None
else:
    run_id = int(slurm_id % num_runs)
    job_id = int(np.floor(slurm_id / num_runs))

f = KineticMonteCarlo(run_id=run_id, sim_id=sim_id, job_id=job_id)
f.generate_two_dim_lattice()
f.simulate()
