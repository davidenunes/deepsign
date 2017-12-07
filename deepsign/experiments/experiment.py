from params import ParamSpace
import csv


class Experiment:
    """ Utility class to help create experiment defining subclasses

        Args:
            name: name for the experiment instance
            param_space: a `ParamSpace` instance
    """

    def __init__(self, name, param_space):
        self.name = name
        self.param_space = param_space
        assert (isinstance(param_space, ParamSpace))


