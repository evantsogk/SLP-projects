from util import EPS, INFINITY, format_arc, calculate_arc_weight


def word_frequencies_dict(fname):
    # read vocabulary and frequencies into a dictionary
    with open(words_path, 'r') as file:
        dictionary = {line.split('\t')[0]: int(line.split('\t')[1].strip('\n')) for line in file.readlines()}

    # convert to percentage frequencies
    total_word_apperances = sum(list(dictionary.values()))
    for key, value in dictionary.items():
        dictionary[key] = value / total_word_apperances
    return dictionary


def make_W_fst(words_freq):
    """
    Create fst with one state that accepts all the words in the vocabulary using the negative logarithm of the word's
    frequency as arc weight
    """
    s = 0  # the only state
    for word, freq in words_freq.items():
        print(format_arc(s, s, word, word, weight=calculate_arc_weight(freq)))
    print(s)


if __name__ == "__main__":
    words_path = '../vocab/words.vocab.txt'

    words_freq = word_frequencies_dict(words_path)
    make_W_fst(words_freq)

