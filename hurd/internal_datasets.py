import os, pickle

from hurd.dataset import Dataset


def load_c13k_data(fb_filter="only_fb"):

    fb_filter = str(fb_filter)

    if fb_filter not in ["only_fb", "no_fb", "None"]:
        raise ValueError("fb_filter must be 'only_fb', 'no_fb', or None")

    fn = "data_c13k_no_amb_" + fb_filter

    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)

    load_path = os.path.join(dir_path, "datasets", fn)

    data = pickle.load(open(load_path,"rb"))

    return data