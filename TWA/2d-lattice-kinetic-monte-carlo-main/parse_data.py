import os
import numpy as np
from datetime import datetime as dt


class Parse:

    def __init__(self, sim_id: str):

        self.sim_id = sim_id

        params_path = os.path.join('k_data', f'{self.sim_id}_params.npy')
        self.params = np.load(params_path, allow_pickle=True)[()]

        self.job_ids = self.params['job_ids']
        self.runs = self.params['simulation_params']['runs']
        self.files_path = os.path.join('k_data_raw', f'{self.sim_id}_raw')

        self.observables = self.params['observables']

        self.save_path = os.path.join('k_data', f'{sim_id}_data.npz')

        if self.job_ids is None:
            self.parse_data()
        else:
            self.parse_data_varied_job_ids()

    def parse_data_varied_job_ids(self):

        data_out = {}

        all_t = None
        all_num_grd = None
        all_num_ryd = None
        all_num_ded = None
        all_ava_time = None
        all_ava_area = None
        all_ava_size = None
        all_runtime = None

        for i, id_ in enumerate(self.job_ids):

            file_stub = f'{self.sim_id}_id{id_}_run'
            files = [f_ for f_ in os.listdir(self.files_path) if f_.startswith(file_stub)]

            if len(files) > 0:
                data_out_at_id = self.get_data(files=files)

                for obs in self.observables:

                    if obs == 'state_dynamics':

                        t = data_out_at_id['t']
                        num_grd = data_out_at_id['num_grd']
                        num_ryd = data_out_at_id['num_ryd']
                        num_ded = data_out_at_id['num_ded']

                        if all_num_grd is None:
                            all_t = t
                            all_num_grd = np.full(shape=(len(t), len(self.job_ids)), fill_value=np.nan, dtype=float)
                            all_num_ryd = np.full(shape=(len(t), len(self.job_ids)), fill_value=np.nan, dtype=float)
                            all_num_ded = np.full(shape=(len(t), len(self.job_ids)), fill_value=np.nan, dtype=float)

                        all_num_grd[:, i] = num_grd
                        all_num_ryd[:, i] = num_ryd
                        all_num_ded[:, i] = num_ded

                    if obs == 'avalanche_exponents':

                        if all_ava_time is None:
                            all_ava_time = np.full(shape=(self.runs, len(self.job_ids)), fill_value=np.nan, dtype=float)
                            all_ava_area = np.full(shape=(self.runs, len(self.job_ids)), fill_value=np.nan, dtype=float)
                            all_ava_size = np.full(shape=(self.runs, len(self.job_ids)), fill_value=np.nan, dtype=float)

                        if len(files) < self.runs:

                            all_ava_time[:len(files), i] = data_out_at_id['avalanche_time']
                            all_ava_area[:len(files), i] = data_out_at_id['avalanche_area']
                            all_ava_size[:len(files), i] = data_out_at_id['avalanche_size']

                        else:

                            all_ava_time[:, i] = data_out_at_id['avalanche_time']
                            all_ava_area[:, i] = data_out_at_id['avalanche_area']
                            all_ava_size[:, i] = data_out_at_id['avalanche_size']

                    if obs == 'runtime_seconds':

                        if all_runtime is None:
                            all_runtime = np.full(shape=(self.runs, len(self.job_ids)), fill_value=np.nan, dtype=float)

                        if len(files) < self.runs:
                            all_runtime[:len(files), i] = data_out_at_id['runtime_seconds']
                        else:
                            all_runtime[:, i] = data_out_at_id['runtime_seconds']

        if 'state_dynamics' in self.observables:

            data_out.update({
                't': all_t,
                'num_grd': all_num_grd,
                'num_ryd': all_num_ryd,
                'num_ded': all_num_ded
            })

        if 'avalanche_exponents' in self.observables:

            data_out.update({
                'avalanche_time': all_ava_time,
                'avalanche_area': all_ava_area,
                'avalanche_size': all_ava_size
            })

        if 'runtime_seconds' in self.observables:

            data_out.update({
                'runtime_seconds': all_runtime,
            })

        print(f'{dt.now().strftime("%Y-%m-%d %H:%M:%S")} - Finished parsing data for {self.sim_id}.', flush=True)

        np.savez(self.save_path, data=data_out, params=self.params)

    def parse_data(self):

        files = os.listdir(self.files_path)
        data_out = self.get_data(files=files)

        print(f'{dt.now().strftime("%Y-%m-%d %H:%M:%S")} - Finished parsing data for {self.sim_id}.', flush=True)

        np.savez(self.save_path, data=data_out, params=self.params)

    def get_data(self, files: list):

        data_out = {}

        for obs in self.observables:

            if obs == 'state_dynamics':

                all_num_grd = None
                all_num_ryd = None
                all_num_ded = None
                t = None

                for i, f_ in enumerate(files):

                    try:
                        data = np.load(os.path.join(self.files_path, f_), allow_pickle=True)['data'][()]

                        if all_num_grd is None:
                            t = data['time']
                            all_num_grd = np.full(shape=(len(t), len(files)), fill_value=np.nan, dtype=float)
                            all_num_ryd = np.full(shape=(len(t), len(files)), fill_value=np.nan, dtype=float)
                            all_num_ded = np.full(shape=(len(t), len(files)), fill_value=np.nan, dtype=float)

                        all_num_grd[:, i] = data['num_grd']
                        all_num_ryd[:, i] = data['num_ryd']
                        all_num_ded[:, i] = data['num_ded']

                    except AttributeError:
                        print(f'WARNING: Loading file {f_} failed.', flush=True)

                all_num_grd = np.nanmean(all_num_grd, axis=1)
                all_num_ryd = np.nanmean(all_num_ryd, axis=1)
                all_num_ded = np.nanmean(all_num_ded, axis=1)

                data_out.update({
                    't': t,
                    'num_grd': all_num_grd,
                    'num_ryd': all_num_ryd,
                    'num_ded': all_num_ded
                })

            if obs == 'avalanche_exponents':

                avalanche_time = np.full(shape=len(files), fill_value=np.nan, dtype=float)
                avalanche_area = np.full(shape=len(files), fill_value=np.nan, dtype=float)
                avalanche_size = np.full(shape=len(files), fill_value=np.nan, dtype=float)

                for i, f_ in enumerate(files):

                    try:
                        data = np.load(os.path.join(self.files_path, f_), allow_pickle=True)['data'][()]

                        try:
                            avalanche_time[i] = data['avalanche_time']
                            avalanche_area[i] = data['avalanche_area']
                            avalanche_size[i] = data['avalanche_size']
                        except KeyError:
                            avalanche_time[i] = data['time_exponent']
                            avalanche_area[i] = data['area_exponent']
                            avalanche_size[i] = data['size_exponent']

                    except AttributeError:
                        print(f'WARNING: Loading file {f_} failed.', flush=True)

                data_out.update({
                    'avalanche_time': avalanche_time,
                    'avalanche_area': avalanche_area,
                    'avalanche_size': avalanche_size
                })

            if obs == 'runtime_seconds':

                runtime_seconds = np.full(shape=len(files), fill_value=np.nan, dtype=float)

                for i, f_ in enumerate(files):

                    try:
                        data = np.load(os.path.join(self.files_path, f_), allow_pickle=True)['data'][()]

                        runtime_seconds[i] = data['runtime_seconds']

                    except AttributeError:
                        print(f'WARNING: Loading file {f_} failed.', flush=True)

                data_out.update({
                    'runtime_seconds': runtime_seconds,
                })

        return data_out


if __name__ == '__main__':
    f = Parse(sim_id='k003c')
