import time, math
from copy import deepcopy
from random import random, gauss
from abc import ABC, abstractmethod

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

from jax import jit, grad
from jax.experimental.optimizers import adam, sgd

from functools import partial

import jax.numpy as jnp
import numpy as np

from hurd.utils import dict2str, fix_jax_dict_floats


class OptimizerBase(ABC):
    def __init__(self):
        self.id = "OptimizerBase"
        self.model = None

        self.best_params = {}
        self.best_loss = np.inf
        self.best_training_loss = np.inf

        self.train_loss_history = []
        self.val_loss_history = []
        self.param_history = []
        self.grad_history = []

    def check_progress(self, ix, n, curr_loss, train_loss, t0=None):

        if self.model.has_validation_data:
            val_result_string = ", Val Loss: {:.5f}".format(curr_loss)
        else:
            val_result_string = ""

        result_str = "[Epoch {}/{}] Train Loss: {:.5f}{}, Elapsed: {:.2f}s".format(
            ix + 1, n, train_loss, val_result_string, time.time() - t0
        )

        if self.tolerance:
            found_improvement = not math.isclose(
                curr_loss, self.best_loss, abs_tol=self.tolerance
            )
            found_improvement = found_improvement and (curr_loss < self.best_loss)
        else:
            found_improvement = True

        if curr_loss < self.best_loss:
            self.best_params = deepcopy(self.model.get_params())
            self.best_loss = curr_loss
            result_str += " * New Best * "

        if train_loss < self.best_training_loss:
            self.best_training_loss = train_loss

        if self.model.verbose > 1:
            print(result_str, flush=True)

        return found_improvement

    def finish(self):
        # report the best model fit and parameters
        if self.model.verbose > -1:
            fstr = "Final best model - Loss: {:.4f}, Params: {}"
            print(fstr.format(self.best_loss, dict2str(self.best_params)))


class GradientBasedOptimizer(OptimizerBase):
    def __init__(
        self,
        alg=sgd,
        lr=0.1,
        n_iters=10,
        use_jit=True,
        tol=None,
        patience=50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id = "GradientBasedOptimizer"
        self.lr = lr
        self.n_iters = n_iters
        self.use_jit = use_jit
        self.batch_size = None
        self.tolerance = tol
        self.patience = patience
        self.alg = alg # update algorithm
        self.alg_args = {} # argument inputs for alg

    def initialize(self, model, batch_size=None):

        self.model = model
        self.batch_size = batch_size

        # get forward pass functions ready for compilation

        def get_train_loss(params):
            self.model.set_params(params)
            return self.model.evaluate(self.model.dataset)

        def get_batch_loss(params, batch):
            self.model.set_params(params)
            return self.model.evaluate(batch)

        if self.model.has_validation_data:

            def get_val_loss(params):
                self.model.set_params(params)
                return self.model.evaluate(self.model.val_dataset)

        else:
            get_val_loss = get_train_loss

        # we take the gradient with respect to either
        # all data at once or one batch at a time

        self.get_train_loss = get_train_loss
        if self.batch_size:
            self.loss_grad = grad(get_batch_loss)
        else:
            self.loss_grad = grad(get_train_loss)
        self.get_val_loss = get_val_loss

        self.opt_init, self.opt_update, self.get_params = self.alg(self.lr, **self.alg_args)
        params = fix_jax_dict_floats(deepcopy(self.model.get_params()))
        self.optimizer_state = self.opt_init(params)
        self.best_params = params

        def jit_step(ix, opt_state, batch):
            current_params = self.get_params(opt_state)

            # take the gradient of the loss function
            if self.batch_size:
                grads = self.loss_grad(current_params, batch)
            else:
                grads = self.loss_grad(current_params)

            # update parameters
            opt_state = self.opt_update(ix, grads, opt_state)
            updated_params = self.get_params(opt_state)

            return opt_state, updated_params, grads

        self.jit_step = jit_step

        # compile everything
        if self.use_jit:
            self.get_train_loss, self.loss_grad, self.get_val_loss, self.jit_step = map(
                jit,
                [self.get_train_loss, self.loss_grad, self.get_val_loss, self.jit_step],
            )

    def step(self, ix):

        t0 = time.time()

        if self.batch_size:

            n_batches = len(self.model.dataset) / self.batch_size

            for (bix, batch) in enumerate(
                tqdm(
                    self.model.dataset.iter_batch(batch_size=self.batch_size),
                    total=n_batches,
                )
            ):
                batch_arrays = batch.as_array(
                    sort=self.model.required_sort, return_targets=True
                )
                batch_dict = {
                    "outcomes": batch_arrays[0],
                    "probabilities": batch_arrays[1],
                    "targets": batch_arrays[2],
                }
                if self.model.requires_dom_mask:
                    batch.generate_dom_mask()
                    batch_dict["dom_mask"] = batch.dom_mask

                # need to keep track of epochs too so
                # this doesn't get reset
                global_index = (ix * n_batches) + bix

                (self.optimizer_state, updated_params, grads) = self.jit_step(
                    global_index, self.optimizer_state, batch_dict
                )

        else:

            (self.optimizer_state, updated_params, grads) = self.jit_step(
                ix, self.optimizer_state, None
            )

        # this is an important line oddly enough.
        # updated_params replaces the traced arrays that
        # jax currently has set as the model params.
        # this allows inference on arbitrary inputs post-training.
        self.model.set_params(updated_params)

        train_loss = np.float(self.get_train_loss(updated_params))
        val_loss = np.float(self.get_val_loss(updated_params))

        # update optimization history
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.param_history.append(fix_jax_dict_floats(updated_params))
        self.grad_history.append(fix_jax_dict_floats(grads))

        # evaluate the model, check for improvement, report speed
        is_improvement = self.check_progress(
            ix, self.n_iters, val_loss, train_loss, t0=t0
        )

        return {
            "epoch": ix,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "improvement": is_improvement,
        }


class SGD(GradientBasedOptimizer):
    def __init__(self, **kwargs):
        super().__init__(alg=sgd, **kwargs)
        self.id = "SGD"


class Adam(GradientBasedOptimizer):
    def __init__(self, b1=0.9, b2=0.999, eps=1e-08, **kwargs):
        super().__init__(alg=adam, **kwargs)
        self.id = "Adam"
        self.alg_args = {"b1": b1, "b2": b2, "eps": eps}
