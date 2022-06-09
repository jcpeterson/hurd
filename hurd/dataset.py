import pickle
import os.path as op
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd


class Gamble:
    def __init__(self, outcomes, probs, n_choices):

        assert len(outcomes) == len(probs)

        self.outcomes = np.array(outcomes)
        self.probs = np.array(probs)
        self.n_choices = n_choices

    def __iter__(self):
        """Generates a tuple of (self.outcomes, self.probs)"""
        for x in [self.outcomes, self.probs]:
            yield x

    def __repr__(self):
        return "Outcomes: {}, Probabilities: {}, # of outcomes: {}".format(
            self.outcomes, self.probs, self.n_choices
        )

    def __eq__(self, other):
        """gambles are equal if they have the same outcomes and probs"""

        # first make sure both gambles are sorted the same way
        sorted_self, sorted_other = self.sort(), other.sort()
        self_outcomes, self_probs = sorted_self
        other_outcomes, other_probs = sorted_other

        if len(self.outcomes) != len(other.outcomes):
            return False

        if np.allclose(self_outcomes, other_outcomes) and np.allclose(
            self_probs, other_probs
        ):
            return True
        else:
            return False

    def sort(self, kind="outcomes_desc", in_place=False):

        if kind not in [
            "outcomes_asc",
            "outcomes_desc",
            "probs_asc",
            "probs_desc",
        ]:
            raise ValueError("Invalid sorting method name passed.")
        else:

            # numpy arrays make sorting easier
            outcomes, probs = map(np.array, [self.outcomes, self.probs])

            if "outcomes" in kind:
                to_sort = outcomes
            else:
                to_sort = probs

            # get indices for sorting
            sort_idx = np.argsort(to_sort)
            # use the preferred order
            if "desc" in kind:
                sort_idx = sort_idx[::-1]

            # perform the sort
            outcomes = list(outcomes[sort_idx])
            probs = list(probs[sort_idx])

            if in_place:
                self.probs = probs
                self.outcomes = outcomes
            else:
                return Gamble(outcomes, probs, self.n_choices)

    def does_dominate(self, other):
        # criterion 1: all other outcomes are less then all self outcomes
        # criterion 2: at least one outcome of self should be higher as opposed to all being equal
        if (np.max(other.outcomes) <= np.min(self.outcomes)) and (
            np.min(other.outcomes) < np.max(self.outcomes)
        ):
            return True
        else:
            return False


class Problem:
    def __init__(self, a_probs, a_outcomes, b_probs, b_outcomes, gamble_id=None):
        self.gamble_id = gamble_id
        n_choices_A, n_choices_B = len(a_outcomes), len(b_outcomes)

        assert len(a_probs) == len(a_outcomes)
        assert len(b_probs) == len(b_outcomes)

        self.A = Gamble(a_outcomes, a_probs, n_choices_A)
        self.B = Gamble(b_outcomes, b_probs, n_choices_B)

    def __iter__(self):
        """Generates a tuple of (self.A, self.B)"""
        for x in [self.A, self.B]:
            yield x

    def as_dict(self):
        return {
            "id": self.gamble_id,
            "A": {
                "probs": self.A.probs,
                "outcomes": self.A.outcomes,
                "n_choices": self.A.n_choices,
            },
            "B": {
                "probs": self.B.probs,
                "outcomes": self.B.outcomes,
                "n_choices": self.B.n_choices,
            },
        }

    def sort(self, kind="outcomes_desc", in_place=False):

        if in_place:
            self.A.sort(kind=kind, in_place=True)
            self.B.sort(kind=kind, in_place=True)
        else:
            a_outcomes, a_probs = self.A.sort(kind=kind, in_place=False)
            b_outcomes, b_probs = self.B.sort(kind=kind, in_place=False)
            return Problem(
                a_probs, a_outcomes, b_probs, b_outcomes, gamble_id=self.gamble_id
            )

    def as_array(self, fixed_dim=9, sort="none"):
        """
        Convert the problem to two dense arrays. Uses zero-padding to ensure a
        fixed dimension.

        Parameters
        ----------
        fixed_dim : int
            The dimensionality of the vector representation for the problem.
            If the number of options in a gamble is less than `fixed_dim`, zero
            padding is added to the right side of the vector.
        sort : str
            Options are outcomes_asc, outcomes_desc, probs_asc, probs_desc,
            indicating whether to sort by outcomes or probabilities in either
            ascending or descending order.

        Returns
        -------
        outcomes : numpy array of shape (2, fixed_dim)
        probabilities : numpy array of shape (2, fixed_dim)
        """

        # optional sorting
        if sort != "none":
            A, B = self.sort(kind=sort, in_place=False)
        else:
            A, B = self.A, self.B

        # make sure we don't truncate any data
        assert len(A.probs) <= fixed_dim
        assert len(B.probs) <= fixed_dim
        assert len(A.outcomes) <= fixed_dim
        assert len(B.outcomes) <= fixed_dim

        # cast to numpy arrays
        a_probs = np.array(A.probs)
        b_probs = np.array(B.probs)
        a_outcomes = np.array(A.outcomes)
        b_outcomes = np.array(B.outcomes)

        # pad arrays
        a_probs = np.pad(a_probs, (0, fixed_dim - len(A.probs)), "constant")
        b_probs = np.pad(b_probs, (0, fixed_dim - len(B.probs)), "constant")
        a_outcomes = np.pad(a_outcomes, (0, fixed_dim - len(A.outcomes)), "constant")
        b_outcomes = np.pad(b_outcomes, (0, fixed_dim - len(B.outcomes)), "constant")

        probs = np.stack([a_probs, b_probs])  # ([A, B], fixed_dim)
        outcomes = np.stack([a_outcomes, b_outcomes])  # ([A, B], fixed_dim)
        return outcomes, probs

    def iterA(self):
        A, nc = self.A, self.A.n_choices
        for out, prob in zip(A.outcomes[:nc], A.probs[:nc]):
            yield (out, prob)

    def iterB(self):
        B, nc = self.B, self.B.n_choices
        for out, prob in zip(B.outcomes[:nc], B.probs[:nc]):
            yield (out, prob)

    def __repr__(self):
        return "Problem {}:\nA: {}\nB: {}".format(self.gamble_id, self.A, self.B)


class Dataset:
    """
    Example for loading the choices13k data:

    >>> from utils import load_choices13k
    >>> gambles, problems, targets = load_choices13k(include_amb=False)
    >>> D = Dataset(dataset_id="choices13k_noAmb")
    >>> D.from_dict(gambles, problems, targets)
    """

    def __init__(self, dataset_id=None):
        self.id = dataset_id
        self._targets = None
        self._problems = OrderedDict()
        self.max_n_outcomes = 9
        self.cached_arrays = None
        self.cached_array_sort = "none"

    def __getitem__(self, p_id):
        target = None if self._targets is None else self._targets[p_id]
        return (self._problems[p_id], target)

    def __setitem__(self, p_id, prob, target=None):
        assert isinstance(prob, Problem)
        self._problems[p_id] = prob

        if target is not None and self._targets is not None:
            self._target[p_id] = target

    def __len__(self):
        return len(self._problems.keys())

    def __iter__(self):
        """Generates a tuple of (p_id, problem, target)"""
        targets = (
            np.array([None] * len(self._problems))
            if self._targets is None
            else list(self._targets.values())
        )
        for p_id, target in zip(self._problems, targets):
            yield p_id, self._problems[p_id], target

    @property
    def problem_ids(self):
        return list(self._problems.keys())

    @property
    def targets(self):
        return None if self._targets is None else np.array(list(self._targets.values()))

    def retarget(self, new_targets):
        # create a copy of this dataset with new targets
        new_dataset = deepcopy(self)
        new_targets_dict = OrderedDict()
        for ix, (p_id, _, _) in enumerate(self):
            new_targets_dict[p_id] = new_targets[ix]
        new_dataset._targets = new_targets_dict
        new_dataset.cache_arrays()
        return new_dataset

    def iloc(self, index):
        p_id = self.problem_ids[index]
        target = None if self._targets is None else self._targets[p_id]
        return (p_id, self._problems[p_id], target)

    def from_dict(self, gamble_dict, prob_ids=None, targets=None, cache_arrays=True):
        if prob_ids is not None:
            assert len(prob_ids) == len(gamble_dict)
        else:
            offset = len(self._problems)
            prob_ids = np.arange(len(gamble_dict)) + offset

        if targets is not None:
            assert len(prob_ids) == len(targets)
            self._targets = OrderedDict()

        for ix, p_id in enumerate(prob_ids):
            entry = gamble_dict[str(ix)]
            a_probs, a_outcomes = zip(*entry["A"])
            b_probs, b_outcomes = zip(*entry["B"])

            a_probs, a_outcomes = map(np.array, [a_probs, a_outcomes])
            b_probs, b_outcomes = map(np.array, [b_probs, b_outcomes])

            self._problems[p_id] = Problem(
                a_probs, a_outcomes, b_probs, b_outcomes, gamble_id=p_id
            )

            if targets is not None:
                self._targets[p_id] = targets[ix]

        # auto cache the array form that models can use
        if cache_arrays:
            self.cache_arrays()

    def from_ordered_dicts(self, problems, targets, cache_arrays=True):
        self._problems = deepcopy(problems)
        self._targets = deepcopy(targets)
        # auto cache the array form that models can use
        if cache_arrays:
            self.cache_arrays()

    def generate_dom_mask(self):

        B_dominated, A_dominated = [], []
        for _, p, _ in self:

            if p.A.does_dominate(p.B):
                B_dominated.append(1)
            else:
                B_dominated.append(0)

            if p.B.does_dominate(p.A):
                A_dominated.append(1)
            else:
                A_dominated.append(0)

        self.dom_mask = (np.array(A_dominated), np.array(B_dominated))

    def as_array(self, fixed_dim=9, sort="none", return_targets=False):
        """
        Convert the dataset to two dense arrays. Uses zero-padding to ensure a
        fixed dimension.

        Parameters
        ----------
        fixed_dim : int
            The dimensionality of the vector representation for each problem.
            If the number of options in a gamble is less than `fixed_dim`, zero
            padding is added to the right side of the vector.
        sort : str
            Options are outcomes_asc, outcomes_desc, probs_asc, probs_desc,
            indicating whether to sort by outcomes or probabilities in either
            ascending or descending order. Just gets passed to
            problem.as_array().

        Returns
        -------
        outcomes : numpy array of shape (n_problems, 2, fixed_dim)
            The reward values. Index i, j, k gives the reward for problem i,
            gamble A if j = 0 else gamble B, outcome number k.
        probabilities : numpy array of shape (n_problems, 2, fixed_dim)
            The outcome probabilities. Index i, j, k gives the probability for
            problem i, gamble A if j = 0 else gamble B, outcome k.
        targets : numpy array of shape (n_problems,)
            The target bRates for each gamble, if available.
        """
        n_probs = len(self._problems)
        outcomes = np.zeros((n_probs, 2, fixed_dim))
        probabilities = np.zeros((n_probs, 2, fixed_dim))

        for ix, (p_id, problem, target) in enumerate(self):
            outcome, probs = problem.as_array(fixed_dim, sort=sort)
            outcomes[ix, ...] += outcome
            probabilities[ix, ...] += probs

        if not return_targets:
            return outcomes, probabilities
        else:
            targets = (
                np.array([None] * n_probs)
                if self._targets is None
                else np.array(list(self._targets.values()))
            )
            return outcomes, probabilities, targets

    def cache_arrays(self, fixed_dim=None, sort="none"):
        if fixed_dim:
            self.cached_arrays = self.as_array(
                fixed_dim=fixed_dim, sort=sort, return_targets=True
            )
        else:
            self.cached_arrays = self.as_array(
                fixed_dim=self.max_n_outcomes, sort=sort, return_targets=True
            )
        self.cached_array_sort = sort

    def load_features(self):

        data_path = op.join(op.split(op.realpath(__file__))[0], "datasets/choices13k/")

        df = pd.read_csv(op.join(data_path, "c13k_obj_feats_uid.csv")).set_index(
            "uniqueID"
        )

        df = df.join(
            pd.read_csv(op.join(data_path, "c13k_naive_feats_uid.csv"))
            .set_index("uniqueID")
            .drop(columns=["Block"])
        )

        df = df.join(
            pd.read_csv(op.join(data_path, "c13k_psych_feats_uid.csv"))
            .set_index("uniqueID")
            .drop(columns=["Block", "diffMins"])
        )

        # only take features for problems in this dataset
        df = df.loc[self.problem_ids, :]

        # bool features to int64
        df.Amb = df.Amb * 1
        df.Feedback = df.Feedback * 1

        df.drop(columns=["Feedback"], inplace=True)

        return df.values

    def iter_batch(self, batch_size=32, shuffle=True, random_state=None):

        if random_state:
            np.random.seed(random_state)

        n_probs = len(self._problems)
        idxs = np.arange(n_probs)

        if shuffle:
            np.random.shuffle(idxs)

        for i_start in range(0, n_probs, batch_size):
            batch_idxs = range(i_start, i_start + batch_size)
            # account for the last, smaller batch
            batch_idxs = [x for x in batch_idxs if x in idxs]
            batch_problems, batch_targets = OrderedDict(), OrderedDict()
            for ix in batch_idxs:
                p_id, problem, target = self.iloc(ix)
                batch_problems[p_id] = problem
                batch_targets[p_id] = target
            batch = Dataset()
            batch.max_n_outcomes = self.max_n_outcomes
            batch.from_ordered_dicts(batch_problems, batch_targets)
            yield batch

    def split(self, p=0.8, n_splits=10, shuffle=True, random_state=None):

        if random_state:
            np.random.seed(random_state)

        n_probs = len(self._problems)
        n_train_probs = int(p * n_probs)

        for split_i in range(n_splits):

            idxs = np.arange(n_probs)
            if shuffle:
                np.random.shuffle(idxs)

            train_idxs = idxs[:n_train_probs]
            val_idxs = idxs[n_train_probs:]

            train_probs, val_probs = OrderedDict(), OrderedDict()
            train_targets, val_targets = OrderedDict(), OrderedDict()

            for ix in idxs:
                p_id, problem, target = self.iloc(ix)
                if ix in train_idxs:
                    train_probs[p_id] = problem
                    train_targets[p_id] = target
                else:
                    val_probs[p_id] = problem
                    val_targets[p_id] = target

            train_data, val_data = map(deepcopy, [self, self])
            train_data._problems = deepcopy(train_probs)
            train_data._targets = train_targets
            val_data._problems = deepcopy(val_probs)
            val_data._targets = val_targets

            train_data.cache_arrays()
            val_data.cache_arrays()

            yield train_data, val_data

    def kfold(self, k=5, shuffle=True, random_state=None):

        if random_state:
            np.random.seed(random_state)

        n_probs = len(self._problems)
        assert n_probs >= k
        split_size = int(np.floor(n_probs / k))

        idxs = np.arange(n_probs)
        if shuffle:
            np.random.shuffle(idxs)

        folds = []
        for fold_i in range(k):
            if fold_i == k - 1:
                # final fold gets the extra int(n_probs % k) data
                folds.append(idxs[fold_i * split_size :])
            else:
                folds.append(idxs[fold_i * split_size : (fold_i + 1) * split_size])

        for fold_i in range(k):

            train_idxs = np.concatenate(
                [fold for j, fold in enumerate(folds) if j != fold_i]
            )
            val_idxs = folds[fold_i]

            train_probs, val_probs = OrderedDict(), OrderedDict()
            train_targets, val_targets = OrderedDict(), OrderedDict()

            for ix in idxs:
                p_id, problem, target = self.iloc(ix)
                if ix in train_idxs:
                    train_probs[p_id] = deepcopy(problem)
                    train_targets[p_id] = target
                else:
                    val_probs[p_id] = deepcopy(problem)
                    val_targets[p_id] = target

            train_data, val_data = map(deepcopy, [self, self])
            train_data._problems = train_probs
            train_data._targets = train_targets
            val_data._problems = val_probs
            val_data._targets = val_targets

            train_data.cache_arrays()
            val_data.cache_arrays()

            yield train_data, val_data

    def intersection(self, other):
        # return a dataset, with arg for ids only
        return set(self.problem_ids).intersection(other.problem_ids)

    def overlaps_with(self, other, deep=False):
        # compare two datasets for overlap
        if not deep:
            if self.intersection(other):
                return True
            else:
                return False
        else:
            # brute compare problem contents
            for _, prob_i, _ in self:
                for _, prob_j, _ in other:

                    n_matched = 0
                    for gamble_i in prob_i:
                        for gamble_j in prob_j:

                            if gamble_i == gamble_j:
                                n_matched += 1
                                break

                        if n_matched > 1:
                            return True
            return False

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))