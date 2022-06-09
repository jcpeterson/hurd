import importlib

from .models.utility_functions import UtilityBase
from .models.weighting_functions import WeightingBase

from .loss_functions import mse, crossentropy

def initializer(f_id, ftype):
    """
    Initialize a utility, loss, or weighting function from a parameter dict.

    Parameters
    ----------
    f_id : str, dict, or object
        A string corresponding to the name of the particular function to
        initialize (e.g., `LinearPWF`), a dict of a function 'id' and
        associated parameter names and values as produced via a `summary`
        method, or an object representing the function itself. In the latter
        case, `initializer` simply returns the passed object with no
        modification. If f_id is a string, the corresponding object is
        initialized using its factory default settings.
    ftype : {"utility", "pwf", "loss"}
        The type of function to initialize.

    Returns
    -------
    fn : function or class instance
        The initialized object
    """
    if ftype not in ["utility", "pwf", "loss"]:
        fstr = "'ftype' must be either 'pwf' or 'utility', but got '{}'"
        raise ValueError(fstr.format(ftype))

    losses = set(["mse", "crossentropy", "ll"])  # valid losses

    if ftype == "utility":
        if isinstance(f_id, str):
            fn = getattr(
                importlib.import_module("hurd.models.utility_functions"), f_id
            )()
        elif isinstance(f_id, UtilityBase):
            fn = f_id
        elif isinstance(f_id, dict):
            fn = getattr(
                importlib.import_module("hurd.models.utility_functions"), f_id["id"]
            )
            fn = fn(**{k: v for k, v in f_id if k not in ["id", "class"]})

    elif "pwf" in ftype:
        if isinstance(f_id, str):
            fn = getattr(
                importlib.import_module("hurd.models.weighting_functions"), f_id
            )()
        elif isinstance(f_id, WeightingBase):
            fn = f_id
        elif isinstance(f_id, dict):
            fn = getattr(
                importlib.import_module("hurd.models.weighting_functions"), f_id["id"]
            )
            fn = fn(**{k: v for k, v in f_id if k not in ["id", "class"]})

    elif "loss" in ftype:
        if f_id == "mse":
            fn = mse
        elif f_id == "crossentropy" or f_id == "ll":
            fn = crossentropy
        else:
            fstr = "Loss must be one of {}, but got '{}'"
            raise ValueError(fstr.format(losses, f_id))
    return fn