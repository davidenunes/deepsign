from exp.params import ParamSpace

# create ParamSpace for baseline experiment
param_space = ParamSpace("baseline.params")
param_space.add_list("num_h", [1, 2])

baseline_exp = param_space
