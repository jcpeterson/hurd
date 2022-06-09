from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.nn import sigmoid

from ..utils import glorot_uniform_init, setup_plotting


class WeightingBase(ABC):
    def __init__(self, **kwargs):
        self.id = "WeightingBase"
        self.parameters = kwargs

    def __str__(self):
        return "{}({})".format(
            self.id, ["{}={}".format(k, v) for k, v in self.parameters.items()]
        )

    def __call__(self, p):
        return self.forward(p)

    @abstractmethod
    def forward(self, p):
        pass

    def set_params(self, param_dict):
        for k, v in param_dict.items():
            if k in self.parameters:
                self.parameters[k] = v

    def summary(self):
        s = {"id": self.id, "class": "pwf"}
        s.update(self.parameters)
        return s

    def apply_fn_excluding_zeros_and_ones(self, probs):
        """ Most probability weighting functions are 
            only applied to 0 > values > 1. This function
            handles this in a jax-compatible way.
        """

        # all probs that are either 0 or 1
        zeros_and_ones_mask = (probs == 0) + (probs == 1)

        # only zeros and ones, all else is also zero
        probs_only_zeros_and_ones = probs * zeros_and_ones_mask

        # now, the opposite, set 0s and 1s to 0
        probs_no_zeros_or_ones = probs * ~zeros_and_ones_mask
        # add 0.5 so all 0/1's are now 0.5
        probs_no_zeros_or_ones += zeros_and_ones_mask * (jnp.ones(probs.shape) * 0.5)

        # apply the probability weighting function
        probs_no_zeros_or_ones = self._forward(probs_no_zeros_or_ones)

        # remove the transformed 0.5 junk
        probs_no_zeros_or_ones *= ~zeros_and_ones_mask

        # add the ones back at the end
        return probs_no_zeros_or_ones + probs_only_zeros_and_ones

    def plot(self, p=None, ax=None, show=False):
        if ax is None:
            plt = setup_plotting()
            ax = plt.subplots()[1]
        if p is None:
            p = np.linspace(0, 1, 100)
        w = self.forward(p)
        ax.plot(p, w)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if show:
            plt.show()


class IdentityPWF(WeightingBase):
    def __init__(self):
        """
        The identity probability weighting function.
        """
        super().__init__()
        self.id = "IdentityPWF"

    def forward(self, p):
        return p


class KT_PWF(WeightingBase):
    def __init__(self, alpha=0.5):
        """
        The Tversky and Kahneman (1992) probability weighting function.

        Notes
        -----
        Tversky and Kahneman's (1992) probability weighting function is given by

            w(p) = p^alpha / ((p^alpha + (1 - p)^alpha)^(1/alpha))

        where `p` is the probability constrained by

            w(0) = 0, w(1) = 1, 0 < p < 1,

        and alpha is the single parameter for the function.

        Observe that `alpha` must be greater than or equal to 0.28, as the function
        is not strictly increasing for `alpha < 0.28`.

        References
        ----------
        [1] Tversky, A., & Kahneman, D. (1992). Advances in prospect theory:
        Cumulative representation of uncertainty. Journal of Risk and Uncertainty,
        5(4), 297-323.

        [2] Wakker, P. P. (2010). Prospect theory: For risk and ambiguity.
        Cambridge, UK: Cambridge University Press. p. 206

        Parameters
        ----------
        alpha : float >= 0.28
            The weighting parameter.
        """
        super().__init__(alpha=alpha)
        self.id = "KT_PWF"

    def forward(self, probs):

        alpha = self.parameters["alpha"]

        # assert alpha >= 0.28, "`alpha` must be greater than 0.28"

        return (probs ** alpha) / (
            (probs ** alpha + (1 - probs) ** alpha) ** (1 / alpha)
        )


class LogOddsLinearPWF(WeightingBase):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        The linear in log odds probability weighting function.

        Notes
        -----
        The linear in log odds probability weighting function is given by

            w(p) = beta * p^alpha / (beta * p^alpha + (1 - p)^alpha),

        where `p` is the probability constrained by

            w(0) = 0, w(1) = 1, 0 < p < 1.

        References
        ----------
        [1] Gonzalez, R., & Wu, G. (1999). On the shape of the probability
        weighting function. Cognitive Psychology, 38, p. 139.

        [2] Wakker, P. P. (2010). Prospect theory: For risk and ambiguity.
        Cambridge, UK: Cambridge University Press. p. 208

        Parameters
        ----------
        alpha : float
        beta : float
        """
        super().__init__(alpha=alpha, beta=beta)
        self.id = "LogOddsLinearPWF"

    def forward(self, probs):

        alpha, beta = self.parameters["alpha"], self.parameters["beta"]

        self._forward = lambda p: (beta * p ** alpha) / (
            beta * p ** alpha + (1 - p) ** alpha
        )

        return self.apply_fn_excluding_zeros_and_ones(probs)


class PowerPWF(WeightingBase):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        The power probability weighting function.

        Notes
        -----
        The power probability weighting function is given by

            w(p) = beta * p^alpha,

        where `p` is the probability constrained by

            w(0) = 0, w(1) = 1, 0 < p < 1.

        References
        ----------
        [1] Stott, H. P. (2006). Cumulative prospect theory's functional menagerie.
        Journal of Risk and Uncertainty, 32(2), 101-130.

        Parameters
        ----------
        alpha : float
        beta : float
        """
        super().__init__(alpha=alpha, beta=beta)
        self.id = "PowerPWF"

    def forward(self, probs):

        alpha, beta = self.parameters["alpha"], self.parameters["beta"]

        self._forward = lambda p: beta * p ** alpha

        return self.apply_fn_excluding_zeros_and_ones(probs)


class NeoAdditivePWF(WeightingBase):
    def __init__(self, alpha=0.4, beta=0.4):
        """
        The neo-additive probability weighting function.

        Notes
        -----
        The neo-additive probability weighting function is given by

            w(p) = beta + alpha * p,

        where `p` is the probability constrained by

            w(0) = 0, w(1) = 1, 0 < p < 1,

        and the two parameters in the function alpha and beta are constrained by

            alpha >= 0
            beta >= 0
            alpha + beta <= 1.

        References
        ----------
        [1] Wakker, P. P. (2010). Prospect theory: For risk and ambiguity.
        Cambridge, UK: Cambridge University Press. Eqn. 7.2.5, p. 208-209

        Parameters
        ----------
        alpha : float >= 0
        beta : float >= 0
        """
        super().__init__(alpha=alpha, beta=beta)
        self.id = "NeoAdditivePWF"

    def forward(self, probs):

        alpha, beta = self.parameters["alpha"], self.parameters["beta"]

        # assert alpha >= 0, "`alpha` must be greater than or equal to 0"
        # assert beta >= 0, "`beta` must be greater than or equal to 0"
        # assert alpha + beta <= 1, "`alpha` + `beta` must be less than or equal to 1"

        self._forward = lambda p: beta + alpha * p

        return self.apply_fn_excluding_zeros_and_ones(probs)


class HyperbolicLogPWF(WeightingBase):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        The hyperbolic-logarithm probability weighting function.

        Notes
        -----
        The hyperbolic-logarithm probability weighting function is given by

            w(p) = (1 - alpha * log(p))^(-beta/alpha)

        where `p` is the probability constrained by

            w(0) = 0, w(1) = 1, 0 < p < 1,

        and the two parameters `alpha` and `beta` are constrained by

            alpha > 0, beta > 0.

        References
        ----------
        [1] Prelec, D. (1998). The probability weighting function. Econometrica,
        60(3), 497-528.

        [2] Luce, R. D. (2001). Reduction invariance and Prelec's weighting
        functions. Journal of Mathematical Psychology, 45(1), p. 176.

        [3] Stott, H. P. (2006). Cumulative prospect theory's functional menagerie.
        Journal of Risk and Uncertainty, 32(2), 101-130. Footnote 3, p. 105.

        Parameters
        ----------
        alpha : float > 0
        beta : float > 0
        """
        super().__init__(alpha=alpha, beta=beta)
        self.id = "HyperbolicLogPWF"

    def forward(self, probs):

        alpha, beta = self.parameters["alpha"], self.parameters["beta"]

        # assert alpha > 0, "`alpha` must be greater than 0"
        # assert beta > 0, "`beta` must be greater than 0"

        self._forward = lambda p: (1 - alpha * jnp.log(p)) ** (-beta / alpha)

        return self.apply_fn_excluding_zeros_and_ones(probs)


class ExponentialPowerPWF(WeightingBase):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        The exponential-power probability weighting function.

        Notes
        -----
        The exponential-power probability weighting function is given by

            w(p) = exp(-alpha/beta * (1-p^beta))

        where `p` is the probability constrained by

            w(0) = 0, w(1) = 1, 0 < p < 1,

        and the two parameters `alpha` and `beta` are constrained by

            alpha != 0
            beta > 0

        References
        ----------
        [1] Prelec, D. (1998). The probability weighting function. Econometrica,
        60(3), 497-528.

        [2] Luce, R. D. (2001). Reduction invariance and Prelec's weighting
        functions. Journal of Mathematical Psychology, 45(1), p. 176.

        [3] Stott, H. P. (2006). Cumulative prospect theory's functional menagerie.
        Journal of Risk and Uncertainty, 32(2), Footnote 3, p. 105.

        Parameters
        ----------
        alpha : float != 0
        beta : float > 0
        """
        super().__init__(alpha=alpha, beta=beta)
        self.id = "ExponentialPowerPWF"

    def forward(self, probs):

        alpha, beta = self.parameters["alpha"], self.parameters["beta"]

        # assert alpha != 0, "`alpha` must not be 0"
        # assert beta > 0, "`beta` must be greater than 0"

        self._forward = lambda p: jnp.exp((-alpha / beta) * (1 - p ** beta))

        return self.apply_fn_excluding_zeros_and_ones(probs)


class CompoundInvariancePWF(WeightingBase):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        The compound invariance probability weighting function.

        Notes
        -----
        The compound invariance probability weighting function is given by

            w(p) = (exp(-beta * (-log(x))^alpha)),

        where `p` is the probability constrained by

            w(0) = 0, w(1) = 1, 0 < p < 1,

        and the two parameters `alpha` and `beta` are constrained by

            alpha > 0
            beta > 0

        References
        ----------
        [1] Prelec, D. (1998). The probability weighting function. Econometrica,
        60(3), 497-528.

        [2] al-Nowaihi, A., & Dhami, S. (2006). A simple derivation of Prelec's
        probability weighting function. Journal of Mathematical Psychology, 50(6),
        521-524.

        [3] Wakker, P. P. (2010). Prospect theory: For risk and ambiguity.
        Cambridge, UK: Cambridge University Press. p. 179, 207.

        Parameters
        ----------
        alpha : float > 0
        beta : float > 0
        """
        super().__init__(alpha=alpha, beta=beta)
        self.id = "CompoundInvariancePWF"

    def forward(self, probs):

        alpha, beta = self.parameters["alpha"], self.parameters["beta"]

        # assert alpha > 0, "`alpha` must be greater than 0"
        # assert beta > 0, "`beta` must be greater than 0"

        self._forward = lambda p: jnp.exp(-beta * ((-jnp.log(p)) ** alpha))

        return self.apply_fn_excluding_zeros_and_ones(probs)


class ConstantRelativeSensitivityPWF(WeightingBase):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        The constant relative sensitivity probability weighting function.

        Notes
        -----
        Constant relative sensitivity probability weighting function is given by

            w(p) = beta^(1-alpha) * p^alpha,

        where `p` is the probability constrained by

            w(0) = 0, w(1) = 1, 0 < p < 1,

        and the two parameters `alpha` and `beta` are constrained by

            alpha > 0
            0 <= beta <= 1

        References
        ----------
        [1] Abdellaoui, M., L'Haridon, O., & Zank, H. (2010). Separating curvature
        and elevation: A parametric probability weighting function. Journal of Risk
        and Uncertainty, 41(1), p. 52.

        Parameters
        ----------
        alpha : float > 0
        beta : float in range [0, 1]
        """
        super().__init__(alpha=alpha, beta=beta)
        self.id = "ConstantRelativeSensitivityPWF"

    def forward(self, probs):

        alpha, beta = self.parameters["alpha"], self.parameters["beta"]

        # assert alpha > 0, "`alpha` must be greater than 0"
        # assert 0 <= beta <= 1, "`beta` must be in range [0, 1]"

        # track probs where p <= beta
        p_leq_beta_mask = probs <= beta
        # set all others to zero so we can replace them later
        p_leq_beta_probs = probs * p_leq_beta_mask
        # weight these probs how they need to be
        p_leq_beta_probs = beta ** (1 - alpha) * p_leq_beta_probs ** alpha
        # once more in case 0s were transformed
        p_leq_beta_probs = probs * p_leq_beta_mask

        # now focus on what we zeroed out before
        p_g_beta_mask = ~p_leq_beta_mask
        # set others to zero
        p_g_beta_probs = probs * p_g_beta_mask
        # now weight these how they need to be (differently)
        p_g_beta_probs = 1 - (1 - beta) ** (1 - alpha) * (1 - p_g_beta_probs) ** alpha
        # once more in case 0s were transformed
        p_g_beta_probs = probs * p_g_beta_mask

        return p_leq_beta_probs + p_g_beta_probs


class NeuralNetworkPWF(WeightingBase):
    def __init__(self, weights=None, biases=None):
        """
        Peterson et al. (2021)
        """

        n_units = 10

        if weights is None:
            weights = []
            weights.append(glorot_uniform_init(n_units, 1))
            weights.append(glorot_uniform_init(n_units, 1))

        if biases is None:
            biases = []
            biases.append(np.zeros(n_units))
            biases.append(np.zeros(1))

        super().__init__(weights=weights, biases=biases)
        self.id = "NeuralNetworkPWF"

    def forward(self, probs):
        w1, w2 = self.parameters["weights"]
        b1, b2 = self.parameters["biases"]

        def nn(prob):
            hidden = sigmoid(jnp.dot(w1, prob) + b1)
            output = sigmoid(jnp.dot(w2, hidden) + b2)
            return output

        nn = vmap(nn)

        orig_shape = probs.shape
        outputs = nn(probs.flatten()).reshape(orig_shape)

        return outputs
