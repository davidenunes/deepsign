from exp.grid import grid_conf, VirtualEnvs

"""
LABMAG Grid Configuration

uses a conda environment pre-loaded with hdf5
along with all other libraries required by the mode
"""
mas = grid_conf(venv=VirtualEnvs.Conda,
                venv_name="deepsign",
                parallel_env="smp",
                num_cores=16)

"""
INGRID Grid Configuration

uses a python virtualenv

both the python and hdf5 modules need to be loaded
before the virtualenv is activated
"""
ingrid = grid_conf(venv=VirtualEnvs.VirtualEnv,
                   venv_root="envs",
                   venv_name="deepsign",
                   parallel_env="mp",
                   num_cores=8,
                   resource_dict={"release": "el7"},
                   queue_name="hpcgrid",
                   module_names=["hdf5-1.8.16", "python-3.5.1"]
                   )
