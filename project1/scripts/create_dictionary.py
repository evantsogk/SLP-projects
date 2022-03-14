import numpy as np


def read_corpus_tokens(fname):
    """ Reads the whole file, splits it into lines and then into tokens.
     """
    with open(fname, 'r') as corpus_file:
        tokens = ' '.join(corpus_file.read().splitlines()).split(' ')
    return tokens


def create_dictionary(tokens):
    """ Returns dictionary with unique tokens as keys and the number of their appearances as values.
    """
    unique_tokens, counts = np.unique(tokens, return_counts=True)

    # ignore tokens with less than 5 appearances
    unique_tokens = unique_tokens[counts >= 5]
    counts = counts[counts >= 5]

    return {unique_tokens[i]: counts[i] for i in range(len(unique_tokens))}


def save_dictionary(fname, dictionary):
    """ Saves the dictionary in vocab/words.vocab.txt
     """
    with open(fname, 'w') as file:
        for key, value in dictionary.items():
            file.write(key + '\t' + str(value) + '\n')


if __name__ == "__main__":
    corpus_path = '../data/gutenberg.txt'
    dict_path = '../vocab/words.vocab.txt'

    tokens = read_corpus_tokens(corpus_path)
    dictionary = create_dictionary(tokens)
    save_dictionary(dict_path, dictionary)

