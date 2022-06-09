import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap
from jax.nn import relu, sigmoid, softmax

from ..decision_model import DecisionModelBase
from .psychophysical import PsychophysicalModel

from ..utils import glorot_uniform_init, setup_plotting
from ..jax_utils import select_array_inputs


class MixtureOfTheories(DecisionModelBase):
    def __init__(
        self,
        variant="full",
        util_func="GeneralPowerLossAverseUtil",
        pwf_func="KT_PWF",
        mixer_units=32,
        models=None,
        include_dom=True,
        inputs="outcomes",
        share_mixers=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id = "MixtureOfSubjectiveFunctions"
        self.n_models = 2
        self.include_dom = include_dom
        if self.include_dom:
            self.requires_dom_mask = True
        self.variant = variant

        self.freeze_network = False
        self.inputs = inputs
        self.share_mixers = share_mixers

        if self.include_dom:
            self.prob_pick_dominated = 0.5

        self.models = {}
        self.models["EU_1"] = PsychophysicalModel(
            util_func=util_func,
            pwf="IdentityPWF",
            optimizer=None,
        )
        self.models["EU_2"] = PsychophysicalModel(
            util_func=util_func,
            pwf="IdentityPWF",
            optimizer=None,
        )
        self.models["SEU_1"] = PsychophysicalModel(
            util_func="IdentityUtil",
            pwf=pwf_func,
            optimizer=None,
        )
        self.models["SEU_2"] = PsychophysicalModel(
            util_func="IdentityUtil",
            pwf="IdentityPWF",
            optimizer=None,
        )

        self.n_models = 2

        input_size = 18 * 2
        if inputs in ["outcomes", "probabilities", "probs"]:
            input_size = 18
        elif inputs == "outcome_count":
            input_size = 1
        elif inputs == "outcome_counts":
            input_size = 2
        self.input_size = input_size

        n_units = mixer_units  # 20
        self.uf_mixer_params = {}
        # to connect inputs (both gambles) to hidden layer
        if input_size > 1:
            self.uf_mixer_params["uf_w1"] = glorot_uniform_init(n_units, input_size)
            self.uf_mixer_params["uf_b1"] = np.zeros(n_units)
        # to hidden layer to the convex weight output layer
        if input_size > 1:
            self.uf_mixer_params["uf_w2"] = glorot_uniform_init(self.n_models, n_units)
            self.uf_mixer_params["uf_b2"] = np.zeros(self.n_models)
        else:
            self.uf_mixer_params["uf_w2"] = glorot_uniform_init(self.n_models, 1)
            self.uf_mixer_params["uf_b2"] = np.zeros(self.n_models)

        # if input_size > 1:
        #     # hack to get hidden activations
        #     def get_hidden(outcome):
        #         hidden = sigmoid(
        #             jnp.dot(self.uf_mixer_params["uf_w1"], outcome)
        #             + self.uf_mixer_params["uf_b1"]
        #         )
        #         return hidden

        # if input_size > 1:
        #     self.get_hidden = vmap(get_hidden)

        def uf_mixer_nn(outcome):
            if input_size > 1:
                hidden = sigmoid(
                    jnp.dot(self.uf_mixer_params["uf_w1"], outcome)
                    + self.uf_mixer_params["uf_b1"]
                )
                output = softmax(
                    jnp.dot(self.uf_mixer_params["uf_w2"], hidden)
                    + self.uf_mixer_params["uf_b2"]
                )
            else:
                output = softmax(
                    jnp.dot(self.uf_mixer_params["uf_w2"], outcome)
                    + self.uf_mixer_params["uf_b2"]
                )
            return output

        self.uf_mixer = vmap(uf_mixer_nn)

        self.pwf_mixer_params = {}
        # to connect inputs (both gambles) to hidden layer
        if input_size > 1:
            self.pwf_mixer_params["pwf_w1"] = glorot_uniform_init(n_units, input_size)
            self.pwf_mixer_params["pwf_b1"] = np.zeros(n_units)
        # to hidden layer to the convex weight output layer
        if input_size > 1:
            self.pwf_mixer_params["pwf_w2"] = glorot_uniform_init(
                self.n_models, n_units
            )
            self.pwf_mixer_params["pwf_b2"] = np.zeros(self.n_models)
        else:
            self.pwf_mixer_params["pwf_w2"] = glorot_uniform_init(self.n_models, 1)
            self.pwf_mixer_params["pwf_b2"] = np.zeros(self.n_models)

        if not self.share_mixers:

            def pwf_mixer_nn(outcome):
                if input_size > 1:
                    hidden = sigmoid(
                        jnp.dot(self.pwf_mixer_params["pwf_w1"], outcome)
                        + self.pwf_mixer_params["pwf_b1"]
                    )
                    output = softmax(
                        jnp.dot(self.pwf_mixer_params["pwf_w2"], hidden)
                        + self.pwf_mixer_params["pwf_b2"]
                    )
                else:
                    output = softmax(
                        jnp.dot(self.pwf_mixer_params["pwf_w2"], outcome)
                        + self.pwf_mixer_params["pwf_b2"]
                    )
                return output

        else:

            def pwf_mixer_nn(outcome):
                if input_size > 1:
                    hidden = sigmoid(
                        jnp.dot(self.uf_mixer_params["uf_w1"], outcome)
                        + self.uf_mixer_params["uf_b1"]
                    )
                    output = softmax(
                        jnp.dot(self.pwf_mixer_params["pwf_w2"], hidden)
                        + self.pwf_mixer_params["pwf_b2"]
                    )
                else:
                    output = softmax(
                        jnp.dot(self.pwf_mixer_params["pwf_w2"], outcome)
                        + self.pwf_mixer_params["pwf_b2"]
                    )
                return output

        self.pwf_mixer = vmap(pwf_mixer_nn)

    def _infer_mixture_weights(self, dataset):
        """ for internal use only"""

        if isinstance(dataset, dict):
            outcomes, probabilities = dataset["outcomes"], dataset["probabilities"]
            if self.include_dom:
                self.A_dominated, self.B_dominated = dataset["dom_mask"]
        else:
            outcomes, probabilities = dataset.as_array(sort=self.required_sort)
            if self.include_dom:
                dataset.generate_dom_mask()
                self.A_dominated, self.B_dominated = dataset.dom_mask

        # the input to the mixer(s) is either one big, flat matrix
        # with all gamble pair information or subset we choose
        mixer_inputs = select_array_inputs(outcomes, probabilities, inputs=self.inputs)

        # infer the convex mixture weights
        uf_convex_weights = self.uf_mixer(mixer_inputs)
        pwf_convex_weights = self.pwf_mixer(mixer_inputs)

        return outcomes, probabilities, uf_convex_weights, pwf_convex_weights

    def infer_mixture_weights(self, dataset):
        """ End user can use this to extract mixture weights """
        _, _, uf_convex_weights, pwf_convex_weights = self._infer_mixture_weights(
            dataset
        )

        return {
            "uf_convex_weights": uf_convex_weights,
            "pwf_convex_weights": pwf_convex_weights,
        }

    def predict(self, dataset):

        (
            outcomes,
            probabilities,
            uf_convex_weights,
            pwf_convex_weights,
        ) = self._infer_mixture_weights(dataset)

        uf1_utils = self.models["EU_1"].utility_fn(outcomes)
        uf2_utils = self.models["EU_2"].utility_fn(outcomes)

        pwf1_weights = self.models["SEU_1"].weight_fn(probabilities)
        pwf2_weights = self.models["SEU_2"].weight_fn(probabilities)

        mixed_utils = (uf1_utils * uf_convex_weights[:, 0].reshape(-1, 1, 1)) + (
            uf2_utils * uf_convex_weights[:, 1].reshape(-1, 1, 1)
        )
        mixed_probs = (pwf1_weights * pwf_convex_weights[:, 0].reshape(-1, 1, 1)) + (
            pwf2_weights * pwf_convex_weights[:, 1].reshape(-1, 1, 1)
        )

        if self.variant == "full":
            mixed_predictions = jnp.sum(mixed_utils * mixed_probs, axis=2)
        elif self.variant == "single_pwf":
            mixed_predictions = jnp.sum(mixed_utils * pwf1_weights, axis=2)
        elif self.variant == "single_uf":
            mixed_predictions = jnp.sum(uf1_utils * mixed_probs, axis=2)
        elif self.variant == "simulate_PT":
            mixed_predictions = jnp.sum(uf1_utils * pwf1_weights, axis=2)

        mixed_predictions = self.decision_function(mixed_predictions)

        if self.include_dom:
            mixed_predictions *= jnp.vstack(
                [1 - self.B_dominated, 1 - self.B_dominated]
            ).T
            mixed_predictions *= jnp.vstack(
                [1 - self.A_dominated, 1 - self.A_dominated]
            ).T
            mixed_predictions += jnp.vstack(
                [
                    self.B_dominated * self.prob_pick_dominated,
                    self.B_dominated * (1 - self.prob_pick_dominated),
                ]
            ).T
            mixed_predictions += jnp.vstack(
                [
                    self.A_dominated * (1 - self.prob_pick_dominated),
                    self.A_dominated * self.prob_pick_dominated,
                ]
            ).T

        return mixed_predictions

    def set_params(self, params):

        for model_key in self.models.keys():
            self.models[model_key].set_params(params[model_key])

        if not self.freeze_network:
            # get the params for the mixer network
            for mixer_params_key in self.uf_mixer_params.keys():
                self.uf_mixer_params[mixer_params_key] = params[mixer_params_key]

            for mixer_params_key in self.pwf_mixer_params.keys():
                self.pwf_mixer_params[mixer_params_key] = params[mixer_params_key]

        if self.include_dom:
            self.prob_pick_dominated = params["prob_pick_dominated"]

        self.T = params["T"]

    def get_params(self):

        params = {}

        # get params for each component model
        for model_key in self.models.keys():
            params[model_key] = self.models[model_key].get_params()

        if not self.freeze_network:
            # get the params for the mixer network
            for mixer_params_key in self.uf_mixer_params.keys():
                params[mixer_params_key] = self.uf_mixer_params[mixer_params_key]

            for mixer_params_key in self.pwf_mixer_params.keys():
                params[mixer_params_key] = self.pwf_mixer_params[mixer_params_key]

        if self.include_dom:
            params["prob_pick_dominated"] = self.prob_pick_dominated

        params["T"] = self.T

        return params

    def plot(self, show=True):

        plt = setup_plotting()

        uf1 = self.models["EU_1"].utility_fn
        uf2 = self.models["EU_2"].utility_fn
        pwf1 = self.models["SEU_1"].weight_fn
        pwf2 = self.models["SEU_2"].weight_fn

        fig, axes = plt.subplots(2, 2)

        uf1_ax, uf2_ax, pwf1_ax, pwf2_ax = (
            axes[0, 0],
            axes[1, 0],
            axes[0, 1],
            axes[1, 1],
        )

        uf1.plot(ax=uf1_ax), uf1_ax.set_title("UF 1")
        uf2.plot(ax=uf2_ax), uf2_ax.set_title("UF 2")
        pwf1.plot(ax=pwf1_ax), pwf1_ax.set_title("PWF 1")
        pwf2.plot(ax=pwf2_ax), pwf2_ax.set_title("PWF 2")

        if show:
            plt.show()
        else:
            return fig, axes
