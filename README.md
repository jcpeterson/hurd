# Human Risky Decision-Making (HURD) Toolkit

**NOTE:** HURD is currently in a pre-release state and thus may contain bugs while being actively developed. We are currently working on more complete documentation. For now, see the quick start guide in the `tutorials` folder.

HURD implements the Mixture of Theories (MOT) model of human risky decision-making along with several other models and baselines (e.g., Prospect Theory) from the paper listed below using Python and jax. It was written by Joshua Peterson and David Bourgin, and is currently a work in progress.

> Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths, T. L. (2021). Using large-scale experiments and machine learning to discover theories of human decision-making. *Science*, *372*(6547), 1209-1214.

BibTeX entry:

```
@article{peterson2021science,
  title={Using large-scale experiments and machine learning to discover theories of human decision-making},
  author={Peterson, Joshua C and Bourgin, David D and Agrawal, Mayank and Reichman, Daniel and Griffiths, Thomas L},
  journal={Science},
  volume={372},
  number={6547},
  pages={1209--1214},
  year={2021},
  publisher={American Association for the Advancement of Science}
}
```

# Basic Usage

```python
from hurd.internal_datasets import load_c13k_data
from hurd.optimizers import Adam
from hurd.models.mixture_of_theories import MixtureOfTheories

# load human data from Peterson et al. (2021)
human_data = load_c13k_data()
# data can be split with a method yeilding a generator
splitter = human_data.split(p=0.9, n_splits=1, shuffle=True, random_state=1)
# here we just want one split
(train_data, val_data) = list(splitter)[0]

# setup an optimizer
optimizer = Adam(lr=0.01, n_iters=2000, use_jit=True)

# initialize a model
model = MixtureOfTheories(
    optimizer=optimizer,
    loss_function="mse",
)

# fit the model parameters
model.fit(dataset=train_data, val_dataset=val_data)
```

More examples can be found in the `tutorials` folder of this repository.

# Requirements

The most important requirement for HURD is JAX, which is used for general comutation, differentiation, and Just In Time (JIT) compilation for speed.

The rest of the requirements are specified in `requirements.txt`.

# License

The data is distributed under the Creative Commons BY-NC-SA 4.0 license. If you intend to use it, please see LICENSE.txt for more information.

# Datasets

HURD includes a copy of risky choice datasets from the paper, but the original release of the data can also be found [here](https://github.com/jcpeterson/choices13k).
