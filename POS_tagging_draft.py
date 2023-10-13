"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Patrick Wang, 2021
"""
from typing import Sequence, Tuple, TypeVar
import numpy as np
import nltk
from collections import Counter

nltk.download("brown")
nltk.download("universal_tagset")


corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
small_subset = corpus[:3]
np.random.seed(42)


def number_of_pos(corpus_sequence: list):
    list_of_pos = []
    for i in range(len(small_subset)):
        pos = small_subset[i][1]
        list_of_pos.append(pos)
        dictionary = dict(Counter(list_of_pos))

    return dictionary


# list_of_words = [sublist[0][0] for sublist in small_subset if sublist]
# print(list_of_words)
def observations(corpus_sequence: list):
    list_of_words = []
    # observations = set()
    observations = []
    dict_unique = {}
    for sublist in small_subset:
        for i in sublist:
            word = i[0]
            list_of_words.append(word)
    index = 0
    for word in list_of_words:
        if word not in observations:
            observations.append(word)
        # observations.add(word)

    index = 0
    for word in observations:
        dict_unique[word] = index
        index += 1

    return dict_unique


def list_of_integers(corpus_sentence: list):
    transition_input = []
    dictionary = observations(corpus_sentence)
    print(dictionary)
    for sublist in corpus_sentence:
        new_sentence = [dictionary[word] for word, pos in sublist]
        transition_input.append(new_sentence)
    return transition_input


print(list_of_integers(small_subset))


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
