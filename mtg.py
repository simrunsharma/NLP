import nltk
from collections import Counter
import random

# Load the corpus
corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
vocabulary = set(corpus)


# Build a list of n-grams from the corpus
def generate_ngrams(corpus, n):
    n_grams = [tuple(corpus[i : i + n]) for i in range(len(corpus) - n + 1)]
    return n_grams


sent = ["robot"]
# sent = ["she", "was", "not"]


def build_dict(corpus, n):
    """Build a dictionary of counts of ngrams."""
    n_grams = generate_ngrams(corpus, n)
    dictionary = Counter(n_grams)
    dictionary = dict(dictionary)

    return dictionary


def get_next_token(sentence, vocabulary, n, corpus, randomize):
    """Return best token coming next in the sequence."""

    if n == 1:
        context = tuple([])
    else:
        context = tuple(sentence[-(n - 1) :])

    n_dictionary = build_dict(corpus, n)
    n_1_dictionary = build_dict(corpus, n - 1)

    prob_dictionary = {}
    if context not in n_1_dictionary:
        return get_next_token(sentence, vocabulary, n - 1, corpus, randomize)

    for token in vocabulary:
        total_gram = context + tuple([token])

        probability = n_dictionary.get(total_gram, 0) / n_1_dictionary[context]

        prob_dictionary[token] = probability

    max_key = [
        "".join(k)
        for k in prob_dictionary
        if prob_dictionary[k] == max(prob_dictionary.values())
    ]

    predicted_gram = (
        max_key[0]
        if not randomize
        else random.choices(
            list(prob_dictionary.keys()), weights=tuple(prob_dictionary.values()), k=1
        )
    )

    if randomize:
        predicted_gram = predicted_gram[0]

    return predicted_gram


############################## FINAL FUNCTION ####################################
def finish_sentence(sentence, n, corpus, randomize):
    while len(sent) < 10:
        predicted_gram = get_next_token(sentence, vocabulary, n, corpus, randomize)
        sent.append(predicted_gram)

        if predicted_gram[0] in [".", "?", "!"]:
            break

    return sent


if __name__ == "__main__":
    x = finish_sentence(sent, 2, corpus, True)
    print(x)
