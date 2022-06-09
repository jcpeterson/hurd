import jax.numpy as jnp

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
        return jnp.sum(((outcomes > 0) * 1) * ((probabilities > 0) * 1), axis=2)
        gamble_pairs_matrix = jnp.sum(gamble_pairs_matrix, axis=1, keepdims=True)
    elif inputs == "outcome_counts":
        return jnp.sum(((outcomes > 0) * 1) * ((probabilities > 0) * 1), axis=2)
    elif inputs == "both":
        return jnp.hstack(
            [
                outcomes.reshape((n_problems, n_outcomes)),
                probabilities.reshape((n_problems, n_outcomes)),
            ]
        )
    else:
        raise ValueError("invalid arg for inputs")

def get_padding_mask(outcomes, probabilities):
    # get a binary mask for where the actual outcomes/probs
    # are located in the 0-padded matrix of (N, gamble, n_outcomes)

    zero_outcomes = (outcomes == 0.0) * 1.0
    zero_probs = (probabilities == 0.0) * 1.0
    padding_mask = zero_outcomes * zero_probs

    return padding_mask


def get_real_data_mask(outcomes, probabilities, extra_mask=None):
    # this locates the actual data without zero-padding
    # just 1 - padding_mask

    padding_mask = get_padding_mask(outcomes, probabilities)
    real_data_mask = 1 - padding_mask

    # in case we want to do any more filtering
    if extra_mask is not None:
        real_data_mask = real_data_mask * extra_mask

    return real_data_mask


def get_outcome_means(outcomes, probabilities, extra_mask=None, keepdims=True):

    # for each gamble in a pair, we want the mean outcome
    # output shape is (N, 2)

    # we need to do some fancy stuff to avoid counting
    # the 0-padding elements
    real_data_mask = get_real_data_mask(outcomes, probabilities, extra_mask=extra_mask)

    outcome_counts = jnp.sum(real_data_mask, axis=2, keepdims=keepdims)
    mean_outcomes = jnp.sum(outcomes, axis=2, keepdims=keepdims) / outcome_counts

    return mean_outcomes


def get_grand_outcome_means(
    outcomes, probabilities, extra_mask=None, keepdim1=True, keepdim2=False
):

    # for each gamble in a pair, we want the mean outcome
    # across both gambles; output shape is (N, 1)

    # we need to do some fancy stuff to avoid counting
    # the 0-padding elements
    real_data_mask = get_real_data_mask(outcomes, probabilities, extra_mask=extra_mask)

    outcome_counts = jnp.sum(real_data_mask, axis=2, keepdims=keepdim1)
    outcome_counts = jnp.sum(outcome_counts, axis=1, keepdims=keepdim2)

    summed_outcomes = jnp.sum(outcomes, axis=2, keepdims=keepdim1)
    summed_outcomes = jnp.sum(summed_outcomes, axis=1, keepdims=keepdim2)

    mean_outcomes = summed_outcomes / outcome_counts

    return mean_outcomes
