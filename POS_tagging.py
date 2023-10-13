"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Patrick Wang, 2021
"""
from typing import Sequence, Tuple, TypeVar
import numpy as np
import nltk
from collections import Counter


corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
small_subset = corpus[:3]


def initial_state_prob(corpus: list):
    """Generate initial state probabilities."""
    pos_dict = {}

    count = 0
    for sentence in corpus:
        pair = sentence[0][1]

        count += 1
        if pair not in pos_dict:
            pos_dict[pair] = 1
        else:
            pos_dict[pair] += 1
    for key in pos_dict:
        pos_dict[key] /= count

    list_of_values = []
    list_of_tags = []
    for i in sorted(pos_dict.keys()):
        list_of_values.append(pos_dict[i])
        list_of_tags.append(i)

    pi = np.array(list_of_values)

    return (pi, list_of_tags)


def transition_matrix(corpus, abc):
    """Building the transition matrix"""

    tag_pair = {}

    for i in corpus:
        for j in range(len(i) - 1):
            tag_pair[(i[j][1], i[j + 1][1])] = (
                tag_pair.get((i[j][1], i[j + 1][1]), 0) + 1
            )

    transition_mat = np.zeros((len(abc), len(abc)), dtype=float)

    for i in tag_pair.keys():
        val_tag = tag_pair[i]
        row_num = abc.index(i[0])
        col_num = abc.index(i[1])
        transition_mat[row_num, col_num] = val_tag

    transition_mat = transition_mat + 1

    transition_mat = transition_mat / transition_mat.sum(axis=1)[:, None]

    return transition_mat


def emission_matrix(corpus, abc):
    """Generate emission dictionary"""

    vocab = []
    for i in corpus:
        for j in i:
            if j[0] not in vocab:
                vocab.append(j[0])

    vocab.append("OOV")

    emission_mat = np.zeros((len(abc), len(vocab)), dtype=float)
    for i in corpus:
        for j in i:
            row_num = abc.index(j[1])
            col_num = vocab.index(j[0])
            emission_mat[row_num, col_num] = emission_mat[row_num, col_num] + 1

    emission_mat = emission_mat + 1

    emission_mat = emission_mat / emission_mat.sum(axis=1)[:, None]

    return emission_mat, vocab


Q = TypeVar("Q")
V = TypeVar("V")


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)


if __name__ == "__main__":
    corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
    (pi, lst_of_tags) = initial_state_prob(corpus)
    A = transition_matrix(corpus, lst_of_tags)
    (B, vocab) = emission_matrix(corpus, lst_of_tags)
    sentences = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]

    word_list = []
    obs = []
    for i in sentences:
        for j in i:
            word_list.append(j[0])

    for i in word_list:
        try:
            ind = vocab.index(i)
        except:
            ind = len(vocab) - 1
        obs.append(ind)

    x, y = viterbi(obs, pi, A, B)

    obs_list_pos = []
    for i in x:
        pos = lst_of_tags[i]
        obs_list_pos.append(pos)

    actual_states = []
    for sentence in sentences:
        for word, tag in sentence:
            actual_states.append(tag)

    equal_list = [(x, y) for x, y in zip(actual_states, obs_list_pos) if x == y]
    non_equal_list = [(x, y) for x, y in zip(actual_states, obs_list_pos) if x != y]

    print(f"Accuracy: {len(equal_list) / len(actual_states) * 100}%")
    print(f"Number of misclassified words: {len(non_equal_list)}")
    print(f"The misclassified words were: {non_equal_list}")
