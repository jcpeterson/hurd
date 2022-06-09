import jax.numpy as jnp


def mse(predictions, targets):
    """Mean squared error loss function"""
    return jnp.mean((predictions[:, 1] - targets) ** 2)


def crossentropy(predictions, targets, eps=1e-30):
    """Crossentropy loss function"""
    # deal with targets being single probabilities.
    # targets are p_choose_B by CPC convention so
    # p_choose_A is 1 - targets
    if len(targets.shape) != 2:
        targets = jnp.vstack([1 - targets, targets]).T
    predictions += eps
    ce_per_datapoint = -jnp.sum(targets * jnp.log(predictions), axis=1)
    return jnp.mean(ce_per_datapoint)
