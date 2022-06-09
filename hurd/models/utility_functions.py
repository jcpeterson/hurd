"""[References]

    Tversky, A., & Kahneman, D. (1992). Advances in prospect theory:
        Cumulative representation of uncertainty.
        Journal of Risk and Uncertainty, 5(4), 297-323.

    Birnbaum, M. H. (2008). New paradoxes of risky decision making.
        Psychological Review, 115(2), 463-501.

    Wakker, P. P. (2010). Prospect theory: For risk and ambiguity.
        Cambridge university press.

    Scholten, M., & Read, D. (2014). Prospect theory and the "forgotten"
        fourfold pattern of risk preferences. Journal of Risk and 
        Uncertainty, DOI 10.1007/s11166-014-9183-2.

    Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., &
        Griffiths, T. L. (2021). Using large-scale experiments and
        machine learning to discover theories of human decision-making.
        Science, 372(6547), 1209-1214.
"""

from abc import ABC, abstractmethod

import numpy as np

from jax import vmap
import jax.numpy as jnp
from jax.nn import sigmoid

from ..utils import glorot_uniform_init, setup_plotting


class UtilityBase(ABC):
    def __init__(self, **kwargs):
        """Base utility function class never used
        as an actual working utility function.
        """
        self.id = "UtilityBase"
        self.parameters = kwargs

    def __str__(self):
        return "()".format(
            self.id, ["=".format(k, v) for k, v in self.parameters.items()]
        )

    def __call__(self, outcome):
        if isinstance(outcome, tuple):
            outcome = jnp.array(outcome)
        return self._forward(outcome)

    def apply_pos_and_neg_fns(self, outcomes):
        """Most utility functions are conditioned on the sign on the outcome.
        This function applies the right functions to the pos/neg outcomes.
        """

        # it's intentional that neither of these include 0
        pos_mask = outcomes > 0.0
        neg_mask = outcomes < 0.0

        # for many functions, u(0) causes errors in backward pass
        # so we need to do u(1) and throw the results out after
        pos_outcomes = self.pos_fn(jnp.where(pos_mask, outcomes, 1.0))
        neg_outcomes = self.neg_fn(jnp.where(neg_mask, outcomes, 1.0))

        pos_outcomes *= pos_mask
        neg_outcomes *= neg_mask

        return pos_outcomes + neg_outcomes

    def plot(self, x=None, ax=None, xlim=(-5, 5), ylim=(-1, 1), show=False):
        if ax is None:
            plt = setup_plotting()
            ax = plt.subplots()[1]
        if x is None:
            x = np.linspace(xlim[0], xlim[1], 100)
        y = self._forward(x)
        ax.plot(x, y)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if show:
            plt.show()

    @abstractmethod
    def _forward(self):
        return NotImplementedError

    def set_params(self, param_dict):
        for k, v in param_dict.items():
            if k in self.parameters:
                self.parameters[k] = v

    def summary(self):
        s = {"id": self.id, "class": "utility"}
        s.update(self.parameters)
        return s


class IdentityUtil(UtilityBase):
    def __init__(self):
        """Identity function (does nothing): U(x) = x"""
        super().__init__()
        self.id = "IdentityUtil"

    def _forward(self, outcomes):
        return outcomes


class LinearUtil(UtilityBase):
    def __init__(self, lambda_=1.0):
        """Plain linear utility: U(x) = lambda * x
        """
        super().__init__(lambda_=lambda_)
        self.id = "LinearUtil"

    def _forward(self, outcomes):
        lambda_ = self.parameters["lambda_"]
        return lambda_ * outcomes


class AsymmetricLinearUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0):
        super().__init__(lambda_=lambda_, alpha=alpha)
        self.id = "AsymmetricLinearUtil"

    def _forward(self, outcomes):
        lambda_ = self.parameters["lambda_"]
        alpha = self.parameters["alpha"]

        # define positive and negative parts of the function
        self.pos_fn = lambda x: alpha * jnp.abs(x)
        self.neg_fn = lambda x: -jnp.abs(lambda_) * jnp.abs(x)

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class LinearLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0):
        """Not to be confused with linear utility in the sense of
        expected value, where utility has a constant slope, the
        "linear" utility function of Tversky & Kahneman (1992)
        is given by

            U(x) = x,              if x >= 0, and

            U(x) = -lambda_ * -x,  if x < 0,

        where U is the utility function, x is the objective outcome
        of a gamble, and lambda is the loss aversion coefficient.

        The following assumption from T. & K. (1992) is made:

            U(-x) = -lambda * U(x) where x >= 0.
        """
        super().__init__(lambda_=lambda_)
        self.id = "LinearLossAverseUtil"

    def _forward(self, outcomes):
        lambda_ = self.parameters["lambda_"]

        # define positive and negative parts of the function
        self.pos_fn = lambda x: jnp.abs(x)
        self.neg_fn = lambda x: -jnp.abs(lambda_) * jnp.abs(x)

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class PowerLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0):
        """
        Implements Wakker (2010), page 78, equation 3.5.1 and Birnbaum (2008), page 466.

        The power utility function is given by

            U(x) = x^alpha,            if x >= 0 and

            U(x) = -lambda * -x^beta,  if x < 0,

        where U is the utility function, x is the objective outcome of a
        gamble, and lambda_ is the loss aversion coefficient.

        Notice that positive and negative values of x are transformed by
        different parameters, allowing asymmetry. This function can implement
        the utility component of prospect thoery.

        The following assumption from T. & K. (1992) is made:

            U(-x) = -lambda * U(x) where x >= 0.

        The implementation is a bit more complicated, and follows:
        https://github.com/cran/pt/
        """
        super().__init__(lambda_=lambda_, alpha=alpha, beta=beta)
        self.id = "PowerLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
        )

        # keep track of strictly positive and strictly negative values
        pos_mask = outcomes > 0
        neg_mask = outcomes < 0  # don't want to include 0

        # for outcomes > 0:
        # first set entries <= 0 to 0
        pos_outcomes = outcomes * pos_mask
        # then add a little epsilon to zeros to avoid errors
        pos_outcomes = pos_outcomes + (~pos_mask * jnp.ones(pos_outcomes.shape))
        # now we can transform them
        # this is a bizarre jax way of doing conditionals:
        # jnp.where(cond, vals_if_true, vals_if_false)
        pos_outcomes = jnp.where(
            alpha > 0,  # if this is true
            pos_outcomes ** alpha,  # return this matrix
            jnp.where(  # else evaluate below block
                alpha == 0, # if this is true
                jnp.log(pos_outcomes), # return this matrix
                1 - (1 + pos_outcomes) ** alpha # else return this one
            ),
        )
        # remove the entries <= 0 that had a little epsilon
        pos_outcomes = pos_outcomes * pos_mask

        # outcomes == 0
        # these should just stay 0 given the above and below

        # for outcomes < 0:
        # first set entries >= 0 to 0
        neg_outcomes = outcomes * neg_mask
        # then add a little -epsilon to zeros to avoid errors
        neg_outcomes = neg_outcomes + (~neg_mask * -jnp.ones(neg_outcomes.shape))
        # now we can transform them
        neg_outcomes = jnp.where(
            beta > 0,
            -lambda_ * (-neg_outcomes) ** beta,
            jnp.where(
                beta == 0,
                -lambda_ * jnp.log(-neg_outcomes),
                -lambda_ * (1.0 - (1.0 - neg_outcomes) ** beta),
            ),
        )
        # remove the entries >= 0 that had a little -epsilon
        neg_outcomes = neg_outcomes * neg_mask

        return pos_outcomes + neg_outcomes


class ExpLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0):
        """
        The exponential utility function.
        
        Wakker (2010), page 80, equation 3.5.4

        The implementation is a bit more complicated, and follows:
        https://github.com/cran/pt/
        """
        super().__init__(lambda_=lambda_, alpha=alpha)
        self.id = "ExpLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha = self.parameters["lambda_"], self.parameters["alpha"]

        # keep track of positive and negative values
        pos_mask = outcomes >= 0
        neg_mask = ~pos_mask

        # first set entries < 0 to 0
        pos_outcomes = outcomes * pos_mask
        # now we can transform them
        pos_outcomes = jnp.where(
            alpha > 0,
            1 - jnp.exp(-alpha * pos_outcomes),
            jnp.where(alpha == 0, pos_outcomes, jnp.exp(-alpha * pos_outcomes) - 1),
        )

        # first set entries >= 0 to 0
        neg_outcomes = outcomes * neg_mask
        # then add a little -epsilon to zeros to avoid errors
        neg_outcomes = neg_outcomes + (~neg_mask * -jnp.ones(neg_outcomes.shape))
        # now we can transform them
        neg_outcomes = jnp.where(
            alpha > 0,
            lambda_ * (1 - jnp.exp(-alpha * -neg_outcomes)),
            jnp.where(
                alpha == 0,
                -lambda_ * neg_outcomes,
                -lambda_ * (jnp.exp(-alpha * -neg_outcomes) - 1),
            ),
        )

        # remove the entries >= 0 that had a little -epsilon
        neg_outcomes = neg_outcomes * neg_mask

        return pos_outcomes + neg_outcomes


class NormExpLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0):
        """
        The normalized exponential utility function is given by

            U(x) = (1 / alpha) * (1 - exp(-alpha * x)),       if x >= 0 and
        
            U(x) = (-lambda / beta) * (1 - exp(-beta * -x)),  if x < 0.

        Scholten, M., & Read, D. (2014), page 71, equation 1
        """
        super().__init__(lambda_=lambda_, alpha=alpha, beta=beta)
        self.id = "NormExpLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
        )

        # define positive and negative parts of the function
        self.pos_fn = lambda x: (1 / alpha) * (1 - jnp.exp(-alpha * jnp.abs(x)))
        self.neg_fn = lambda x: (-jnp.abs(lambda_) / beta) * (
            1 - jnp.exp(-beta * jnp.abs(x))
        )

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class NormLogLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0):
        """
        The normalized logarithmic utility function is given by
                
            U(x) = (1 / alpha) * log(1 + alpha * x),       if x >= 0 and

            U(x) = (-lambda / beta) * log(1 + beta * -x),  if x < 0.

        Scholten, M., & Read, D. (2014), page 71, equation 2
        """
        super().__init__(lambda_=lambda_, alpha=alpha, beta=beta)
        self.id = "NormLogLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
        )

        # define positive and negative parts of the function
        self.pos_fn = lambda x: (1 / alpha) * jnp.log(1 + alpha * jnp.abs(x))
        self.neg_fn = lambda x: (-jnp.abs(lambda_) / beta) * jnp.log(
            1 + beta * jnp.abs(x)
        )

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class NormPowerLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0):
        """
        The normalized power utility function is given by
                
            U(x) = (1 / (1 + alpha)) * x^(1 / (1 + alpha)),       if x >= 0 and

            U(x) = (-lambda / (1 + beta)) * -x^(1 / (1 + beta)),  if x < 0.

        Scholten, M., & Read, D. (2014), page 71, equation 3
        """
        super().__init__(lambda_=lambda_, alpha=alpha, beta=beta)
        self.id = "NormPowerLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
        )

        # define positive and negative parts of the function
        self.pos_fn = lambda x: (1 / 1 + alpha) * (jnp.abs(x) ** (1 / 1 + alpha))
        self.neg_fn = lambda x: (-(jnp.abs(lambda_) / 1 + beta)) * (
            jnp.abs(x) ** (1 / 1 + beta)
        )

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class QuadLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0):
        """
        U(x) = x - alpha ** x^2,                if x >= 0 and

        U(x) = -lambda * (-x) - beta * (-x)^2,  if x < 0.
        """
        super().__init__(lambda_=lambda_, alpha=alpha, beta=beta)
        self.id = "QuadLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
        )

        # lambda doesn't need to be negated here
        self.pos_fn = lambda x: jnp.abs(x) - (alpha * (jnp.abs(x) ** 2))
        self.neg_fn = lambda x: jnp.abs(lambda_) * (
            jnp.abs(x) - (beta * (jnp.abs(x) ** 2))
        )

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class LogLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0):
        super().__init__(lambda_=lambda_, alpha=alpha, beta=beta)
        self.id = "LogLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
        )

        # define positive and negative parts of the function
        self.pos_fn = lambda x: jnp.log(alpha + jnp.abs(x))
        self.neg_fn = lambda x: -jnp.abs(lambda_) * (jnp.log(beta + jnp.abs(x)))

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class ExpPowerLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__(lambda_=lambda_, alpha=alpha, beta=beta, gamma=gamma)
        self.id = "ExpPowerLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta, gamma = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
            self.parameters["gamma"],
        )

        # define positive and negative parts of the function
        self.pos_fn = lambda x: gamma - jnp.exp(-beta * jnp.abs(x) ** alpha)
        self.neg_fn = lambda x: -jnp.abs(lambda_) * gamma - jnp.exp(
            -beta * jnp.abs(x) ** alpha
        )

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class GeneralLinearLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0):
        super().__init__(lambda_=lambda_, alpha=alpha, beta=beta)
        self.id = "GeneralLinearLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
        )

        # define positive and negative parts of the function
        self.pos_fn = lambda x: alpha * x
        self.neg_fn = lambda x: -jnp.abs(lambda_) * beta * x

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class GeneralPowerLossAverseUtil(UtilityBase):
    def __init__(self, lambda_=1.0, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        """
        The general power utility function is given by:

            U(x) = beta * x^alpha,                if x >= 0 and

            U(x) = -lambda * (delta * -x)^gamma,  if x < 0,

        where U is the utility function, x is the objective outcome
        of a gamble, and lambda is the loss aversion coefficient.

        The following assumption from T. & K. (1992) is made:

            U(-x) = -lambda * U(x) where x >= 0.

        References:
         
            Tversky, A., & Kahneman, D. (1992), page 309 
            Birnbaum, M. H. (2008), page 466, equation 2
        """
        super().__init__(
            lambda_=lambda_, alpha=alpha, beta=beta, gamma=gamma, delta=delta
        )
        self.id = "GeneralPowerLossAverseUtil"

    def _forward(self, outcomes):
        lambda_, alpha, beta, gamma, delta = (
            self.parameters["lambda_"],
            self.parameters["alpha"],
            self.parameters["beta"],
            self.parameters["gamma"],
            self.parameters["delta"],
        )

        # define positive and negative parts of the function
        self.pos_fn = lambda x: beta * jnp.abs(x) ** alpha
        self.neg_fn = lambda x: -jnp.abs(lambda_) * ((delta * jnp.abs(x)) ** gamma)

        # apply utility function in jax-compatible way
        return self.apply_pos_and_neg_fns(outcomes)


class NeuralNetworkUtil(UtilityBase):

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
        self.id = "NeuralNetworkUtil"

    def _forward(self, outcomes):
        w1, w2 = self.parameters["weights"]
        b1, b2 = self.parameters["biases"]

        def nn(outcome):
            # hidden = jnp.tanh(jnp.dot(w1, outcome) + b1)
            hidden = sigmoid(jnp.dot(w1, outcome) + b1)
            # output = sigmoid(jnp.dot(w2, hidden) + b2)
            output = jnp.dot(w2, hidden) + b2
            return output

        nn = vmap(nn)

        orig_shape = outcomes.shape
        outputs = nn(outcomes.flatten()).reshape(orig_shape)

        return outputs
