import os
from itertools import product

import numpy as np


def setup_plotting():
    try:
        import matplotlib.pyplot as plt
        from IPython.core.getipython import get_ipython
    except:
        raise ImportError("Can't find matplotlib.")
    try:
        ipython = get_ipython()
        ipython.run_line_magic("matplotlib", "inline")
    except:
        print("Can't detect a jupyter notebook. Plots will not render inline.")
    return plt


def glorot_uniform_init(nout, nin):
    """ Glorot uniform parameter initializer """
    sd = np.sqrt(6.0 / (nin + nout))
    return np.random.uniform(-sd, sd, size=(nout, nin)).squeeze()


def select_array_inputs(outcomes, probabilities, inputs="both"):

    # number of observations/problems in the dataset
    n_problems = outcomes.shape[0]
    # out many max outcomes per gamble used for the array
    n_outcomes = outcomes.shape[1] * outcomes.shape[2]

    # flatten the input data (n_samples, (input_shape)) --> (n_samples, 20 or 40)
    if inputs == "outcomes":
        return outcomes.reshape((n_problems, n_outcomes))
    elif inputs in ["probabilities", "probs"]:
        return probabilities.reshape((n_problems, n_outcomes))
    elif inputs == "outcome_count":
        return np.sum(((outcomes > 0) * 1) * ((probabilities > 0) * 1), axis=2)
        gamble_pairs_matrix = np.sum(gamble_pairs_matrix, axis=1, keepdims=True)
    elif inputs == "outcome_counts":
        return np.sum(((outcomes > 0) * 1) * ((probabilities > 0) * 1), axis=2)
    elif inputs == "both":
        return np.hstack(
            [
                outcomes.reshape((n_problems, n_outcomes)),
                probabilities.reshape((n_problems, n_outcomes)),
            ]
        )
    else:
        raise ValueError("invalid arg for inputs")


def fix_jax_dict_floats(dict_):
    new_dict_ = {}
    for key, val in dict_.items():
        try:
            fixed_val = float(val)
            new_dict_[key] = fixed_val
        except:
            new_dict_[key] = dict_[key]
    return new_dict_


def mkdir(fp):
    """Create the intermediate dirs in `fp` if they do not already exist"""
    if not os.path.isdir(fp):
        os.makedirs(fp)
    return fp


def float2str(float_, precision=2):
    return "{0:.2f}".format(round(float(float_), precision))


def keyval2str(k, v):
    return "{}: {}".format(k, float2str(v))


def dict2str(dict_):
    string = ""
    for k, v in dict_.items():
        string += "{} | ".format(keyval2str(k, v))
    return string


def flatten_grid(grid):
    """
    Flatten `grid` (a dictionary of `parameter_id: value_list`s) into a list of
    parameter combinations. Each parameter combo is a dict of parameter_id:
    single value.

    Based on scikit-learn's `ParameterGrid` class.
    """
    flat_grid = []

    # sort the keys of a dictionary for reproducibility
    items = sorted(grid.items())

    keys, values = zip(*items)
    for v in product(*values):
        params = dict(zip(keys, v))
        flat_grid.append(params)

    return flat_grid


def list_utility_fn_class_names():
    return [
        "IdentityUtil",
        "LinearUtil",
        "AsymmetricLinearUtil",
        "LinearLossAverseUtil",
        "PowerLossAverseUtil",
        "ExpLossAverseUtil",
        "NormExpLossAverseUtil",
        "NormLogLossAverseUtil",
        "NormPowerLossAverseUtil",
        "QuadLossAverseUtil",
        "LogLossAverseUtil",
        "ExpPowerLossAverseUtil",
        "GeneralLinearLossAverseUtil",
        "GeneralPowerLossAverseUtil",
        "NeuralNetworkUtil",
    ]
