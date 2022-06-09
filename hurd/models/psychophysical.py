import numpy as np

import jax.numpy as jnp
from jax import vmap

from ..decision_model import DecisionModelBase

from ..initializer import initializer


class PsychophysicalModel(DecisionModelBase):
    def __init__(self, util_func="GeneralPowerLossAverseUtil", pwf="KT_PWF", **kwargs):
        super().__init__(**kwargs)
        self.id = "PsychophysicalModel"

        # initialize the utility, probability weighting, and loss parameters
        self.utility_fn = initializer(util_func, "utility")
        self.weight_fn = initializer(pwf, "pwf")

    def get_params(self):
        params = {}
        for key in self.utility_fn.parameters.keys():
            params["uf_" + key] = self.utility_fn.parameters[key]

        for key in self.weight_fn.parameters.keys():
            params["pwf_" + key] = self.weight_fn.parameters[key]

        # either get softmax temperature
        if self.stochastic_spec == "softmax":
            params["T"] = self.T
        # or get constant error term
        elif self.stochastic_spec == "constant-error":
            params["mu"] = self.mu

        return params

    def set_params(self, params):
        # set utility function params
        uf_params = {k[3:]: v for k, v in params.items() if k.startswith("u")}
        self.utility_fn.set_params(uf_params)

        pwf_params = {k[4:]: v for k, v in params.items() if k.startswith("p")}
        self.weight_fn.set_params(pwf_params)

        # either set softmax temperature
        if self.stochastic_spec == "softmax":
            self.T = params["T"]
        # or set constant error term
        elif self.stochastic_spec == "constant-error":
            self.mu = params["mu"]

    def predict(self, dataset):

        if isinstance(dataset, dict):
            outcomes, probabilities = dataset["outcomes"], dataset["probabilities"]
        else:
            outcomes, probabilities = dataset.as_array(sort=self.required_sort)

        U, W = self.utility_fn, self.weight_fn

        outcomes, probabilities = U(outcomes), W(probabilities)

        utils = jnp.sum(outcomes * probabilities, axis=2)

        return self.decision_function(utils)


class ExpectedUtilityModel(PsychophysicalModel):
    # will just be a wrapper with a fixed linear pwf
    # and takes only a specified utility function
    def __init__(self, util_func="GeneralPowerLossAverseUtil", **kwargs):
        super().__init__(
            util_func=util_func, pwf="IdentityPWF", **kwargs
        )
        self.id = "ExpectedUtilityModel"


class ExpectedValueModel(ExpectedUtilityModel):
    # will just be a wrapper with a fixed linear pwf
    def __init__(self, **kwargs):
        super().__init__(util_func="IdentityUtil", **kwargs)
        self.id = "ExpectedValueModel"


class ProspectTheoryModel(PsychophysicalModel):
    def __init__(self, util_func="GeneralPowerLossAverseUtil", pwf="KT_PWF", **kwargs):
        super().__init__(util_func=util_func, pwf=pwf, **kwargs)
        self.id = "ProspectTheoryModel"


class CumulativeProspectTheoryModel(PsychophysicalModel):
    def __init__(self, pwf_pos="KT_PWF", pwf_neg="KT_PWF", **kwargs):
        super().__init__(**kwargs)
        self.id = "CumulativeProspectTheoryModel"

        # we sort outcomes/probs based on ascending outcome value,
        # following Fennema & Wakker (1997)
        self.required_sort = "outcomes_asc"

        # initialize the two separate probability weighting functions
        self.weight_fn_pos = initializer(pwf_pos, "pwf")
        self.weight_fn_neg = initializer(pwf_neg, "pwf")

        # no single pwf for this class
        self.weight_fn = None

    def value_per_gamble(self, probs, outcomes, ld_mask):

        WP, WN = self.weight_fn_pos, self.weight_fn_neg

        def get_val():
            ld_p_cumsum = WN(jnp.clip(jnp.cumsum(probs), 0, 1))
            ld_p = jnp.hstack([ld_p_cumsum[0], ld_p_cumsum[1:] - ld_p_cumsum[:-1]])
            ld_p *= outcomes * ld_mask
            return ld_p

        ld_p = jnp.where(ld_mask[0] != 0, get_val(), 0)

        gd_mask = 1 - ld_mask
        gd_p_cumsum = WP(jnp.clip(jnp.cumsum(probs * gd_mask), 0, 1))
        gd_p = jnp.hstack([gd_p_cumsum[0], gd_p_cumsum[1:] - gd_p_cumsum[:-1]])
        gd_p *= outcomes * gd_mask

        return jnp.sum(ld_p) + jnp.sum(gd_p)

    def predict(self, dataset):

        if isinstance(dataset, dict):
            outcomes, probabilities = dataset["outcomes"], dataset["probabilities"]
        else:
            outcomes, probabilities = dataset.as_array(sort=self.required_sort)

        # most of the following to find first positive outcomes
        # do all searchsorts in advance
        def searchsort(x):
            return [np.searchsorted(x[0], 0), np.searchsorted(x[1], 0)]

        ld_masks = []
        for i in range(outcomes.shape[0]):
            first_pos = searchsort(outcomes[i])
            ld_masks.append(
                [
                    np.hstack(
                        [
                            np.ones(first_pos[0]),
                            np.zeros(outcomes.shape[2] - first_pos[0]),
                        ]
                    ),
                    np.hstack(
                        [
                            np.ones(first_pos[1]),
                            np.zeros(outcomes.shape[2] - first_pos[1]),
                        ]
                    ),
                ]
            )

        ld_masks = np.array(ld_masks)

        # pre-apply utility function
        outcomes = self.utility_fn(outcomes)

        value_per_gamble = vmap(self.value_per_gamble)

        gambleA_values = value_per_gamble(
            probabilities[:, 0], outcomes[:, 0], ld_masks[:, 0]
        )

        gambleB_values = value_per_gamble(
            probabilities[:, 1], outcomes[:, 1], ld_masks[:, 1]
        )

        return self.decision_function(jnp.vstack([gambleA_values, gambleB_values]).T)

    def get_params(self):
        params = {}
        for key in self.utility_fn.parameters.keys():
            params["uf_" + key] = self.utility_fn.parameters[key]

        for key in self.weight_fn_pos.parameters.keys():
            params["pos_pwf_" + key] = self.weight_fn_pos.parameters[key]

        for key in self.weight_fn_neg.parameters.keys():
            params["neg_pwf_" + key] = self.weight_fn_neg.parameters[key]

        # either get softmax temperature
        if self.stochastic_spec == "softmax":
            params["T"] = self.T
        # or get constant error term
        elif self.stochastic_spec == "constant-error":
            params["mu"] = self.mu

        return params

    def set_params(self, params):
        # set utility function params
        uf_params = {k[3:]: v for k, v in params.items() if k.startswith("u")}
        self.utility_fn.set_params(uf_params)

        # set the probability weighting function parameters
        pwf_pos = {k[8:]: v for k, v in params.items() if k.startswith("p")}
        pwf_neg = {k[8:]: v for k, v in params.items() if k.startswith("n")}
        self.weight_fn_pos.set_params(pwf_pos)
        self.weight_fn_neg.set_params(pwf_neg)

        # either set softmax temperature
        if self.stochastic_spec == "softmax":
            self.T = params["T"]
        # or set constant error term
        elif self.stochastic_spec == "constant-error":
            self.mu = params["mu"]


class TransferOfAttentionExchangeModel(DecisionModelBase):
    def __init__(self, **kwargs):
        """
        "Special" TAX model from:

            Birnbaum, M. H. (2008). New paradoxes of risky 
            decision making. Psychological Review, 115, 463–501.
        """
        super().__init__(**kwargs)
        self.id = "TransferOfAttentionExchange"

        # for TAX, otucomes must be ordered smallest to largest
        self.required_sort = "outcomes_asc"

        # single parameter for the power weighting function
        self.gamma = 0.5
        # single parameter for the power utility function
        self.beta = 0.5
        # configural weight parameter
        self.delta = 0.0

    def set_params(self, params):
        self.gamma = params["gamma"]
        self.beta = params["beta"]
        self.delta = params["delta"]

        self.T = params["T"]

    def get_params(self):
        params = {}
        params["gamma"] = self.gamma
        params["beta"] = self.beta
        params["delta"] = self.delta

        params["T"] = self.T
        return params

    def get_constraints(self):
        raise NotImplementedError

    def predict(self, dataset):

        if isinstance(dataset, dict):
            outcomes, probabilities = dataset["outcomes"], dataset["probabilities"]
        else:
            outcomes, probabilities = dataset.as_array(sort=self.required_sort)

        # apply power utility function
        signs = jnp.sign(outcomes)
        absolute_outcomes = jnp.abs(outcomes)
        utilities = signs * absolute_outcomes ** self.beta

        # apply power probability weighting function
        weighted_probs = probabilities ** self.gamma

        # first part of model is just subjective expected utility
        seu = jnp.sum(utilities * weighted_probs, axis=2)

        # takes a single gamble and returns the TAX term
        def per_gamble_TAX_sum(outcomes, probs):

            # array length
            n = outcomes.size

            # mask for the real outcomes (excludes zero padding)
            non_padding_mask = ~((probs == 0.0) & (outcomes == 0.0)) * 1.0
            # actual number of outcomes without zero-padding
            m = jnp.sum(non_padding_mask)
            # helps remove the bad terms later due to the zero padding
            remove_mask = jnp.outer(non_padding_mask, jnp.ones(n - 1))[1:]

            # i and j outcomes in the tax model formula
            x_i, x_j = outcomes[1:], outcomes[:-1]

            x_i_outer = jnp.outer(x_i, jnp.ones(n - 1))
            x_j_outer = jnp.outer(x_j, jnp.ones(n - 1)).T

            xij_diffs = x_i_outer - x_j_outer
            # make sure to remove useless upper echelon
            xij_diffs = xij_diffs * jnp.triu(jnp.ones((n - 1, n - 1))).T

            # i and j probabilities in the tax model formula
            p_i, p_j = probs[1:], probs[:-1]

            # calculate both possible weights,
            # to be conditioned on delta
            pi_weights = (p_i * self.delta) / (m + 1)
            pj_weights = (p_j * self.delta) / (m + 1)

            # need to make them align with xij_diffs
            p_i_outer = jnp.outer(pi_weights, jnp.ones(n - 1))
            p_j_outer = jnp.outer(pj_weights, jnp.ones(n - 1)).T

            # selecting one weight set or the other
            delta_is_neg = (self.delta < 0) * 1.0
            weights = p_i_outer * delta_is_neg + p_j_outer * (1 - delta_is_neg)

            tax_sum_terms = xij_diffs * weights * remove_mask

            tax_sum = jnp.sum(tax_sum_terms)

            return tax_sum

        per_gamble_TAX_sum = vmap(per_gamble_TAX_sum)

        gambleA_tax_sums = per_gamble_TAX_sum(weighted_probs[:, 0], utilities[:, 0])
        gambleB_tax_sums = per_gamble_TAX_sum(weighted_probs[:, 1], utilities[:, 1])

        tax_sums = jnp.vstack([gambleA_tax_sums, gambleB_tax_sums]).T

        final_utils = (seu + tax_sums) / jnp.sum(weighted_probs, axis=2)

        return self.decision_function(final_utils)
