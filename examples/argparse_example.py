import argparse
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2int(v):
    return int(float(v))



parser = argparse.ArgumentParser(description="LBL base experiment")


# clean argparse a bit
def param(name, argtype, default, valid=None):
    if valid is not None:
        parser.add_argument('-{}'.format(name), dest=name, type=argtype, default=default, choices=valid)
    else:
        parser.add_argument('-{}'.format(name), dest=name, type=argtype, default=default)


parser = argparse.ArgumentParser(description="testing argparse")

param("run",str2int,1)

args = parser.parse_args(["-run","2.3"])


arg_dict = vars(args)

print(arg_dict)
