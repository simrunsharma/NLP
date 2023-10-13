"""Pytorch."""
import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def logit(x: FloatArray) -> FloatArray:
    """Compute logit (inverse sigmoid)."""
    return np.log(x) - np.log(1 - x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vector so that it sums to 1."""
    return x / torch.sum(x)


def loss_fn(p: float) -> float:
    """Compute loss to maximize probability."""
    return -p


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        s0 = logit(np.ones((V, 1)) / V)
        self.s = nn.Parameter(torch.tensor(s0.astype("float32")))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        p = normalize(torch.sigmoid(self.s))

        return torch.sum(input, 1, keepdim=True).T @ torch.log(p), p


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype("float32"))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 200
    learning_rate = 0.1

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_array = []
    i_array = []
    forward3_values = []

    for _ in range(num_iterations):
        p_pred, probabilities = model(x)

        loss = -p_pred
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        i_array.append(_)
        loss_array.append(loss.detach().numpy()[0])

    prob_dict = give_dictionary()

    forward3_value = forward_3(prob_dict, x)
    forward3_values.append(forward3_value.item())

    plt.figure(figsize=(8, 6))
    plt.plot(
        i_array,
        loss_array,
        # forward3_values,
        label="Model Loss",
    )
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.title("Loss vs. Number of Iterations")
    plt.grid(True)
    plt.savefig("output.png")
    plt.axhline(forward3_values, color="r", linestyle="--", label="Theoretical Loss")
    plt.legend()
    the_dictionary = give_dictionary()
    prob_dict = create_orderedDict(the_dictionary)

    values_tensor = torch.tensor(list(prob_dict.values()), dtype=torch.float32)
    reshaped_tensor = values_tensor.view(28, 1)
    vocabulary_2 = [chr(i + ord("a")) for i in range(26)] + [" ", "None"]

    fig, ax = plt.subplots()
    shapes = ["^", "x"]
    probabilities.detach().numpy()
    ax.scatter(
        vocabulary_2,
        probabilities.detach().numpy(),
        marker=shapes[0],
        label="Model Probabilities",
    )
    ax.scatter(
        vocabulary_2,
        reshaped_tensor.numpy(),
        marker=shapes[1],
        label="Theoretical Probabilities",
    )
    ax.legend(loc="upper left")
    plt.xlabel("Vocabulary")
    plt.show()


def give_dictionary():
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()
    tokens = [char for char in text]
    count_dict = []
    prob_dict = {}
    valid_vocab = [chr(i + ord("a")) for i in range(26)] + [" ", None]
    non_valid_prob = 0.0

    count = Counter(text)

    cummulative_counts = {}
    summ = 0

    for key, value in count.items():
        if key in valid_vocab:
            probability = value / sum(count.values())
            prob_dict[key] = probability
        else:
            summ = summ + value
        prob_dict["None"] = summ / sum(count.values())
    keys_to_delete = [key for key, value in prob_dict.items() if value is None]

    # Delete the key-value pairs with None values
    for key in keys_to_delete:
        del prob_dict[key]
    return prob_dict


def custom_sort(item):
    key, value = item
    # Sort by moving " " and "None" to the end while preserving their original order
    if key == " ":
        return (1, key)
    elif key == "None":
        return (2, key)
    else:
        return (0, key)


def create_orderedDict(dict):
    sorted_dict = sorted(dict.items(), key=custom_sort)
    ordered_version = OrderedDict(sorted_dict)
    return ordered_version


def forward_3(dictionary: dict, input: torch.Tensor) -> torch.Tensor:
    the_dictionary = give_dictionary()
    prob_dict = create_orderedDict(the_dictionary)

    values_tensor = torch.tensor(list(prob_dict.values()), dtype=torch.float32)
    reshaped_tensor = values_tensor.view(28, 1)

    result = torch.sum(input, 1, keepdim=True).T @ torch.log(reshaped_tensor)
    result_loss = -result
    return result_loss


if __name__ == "__main__":
    gradient_descent_example()
