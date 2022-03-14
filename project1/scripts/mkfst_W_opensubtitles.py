from util import EPS, INFINITY, format_arc, calculate_arc_weight


def make_W_fst(syms_file, freq_file):
    """
    Create fst with one state that accepts all the words in the vocabulary using the negative logarithm of the word's
    frequency as arc weight
    """
    # read tokens
    with open(syms_file, 'r') as file:
        tokens = [line.split('\t')[0] for line in file.readlines()]
        tokens.remove(EPS)

    # create word-frequencies dictionary (unpreprocessed)
    with open(freq_file, 'r', encoding="utf-8") as file:
        dictionary = {line.split(' ')[0]: int(line.split(' ')[1]) for line in file.readlines()}

    # convert to percentage frequencies
    total_word_apperances = sum(list(dictionary.values()))
    for key, value in dictionary.items():
        dictionary[key] = value / total_word_apperances

    s = 0  # the only state
    for word in tokens:
        print(format_arc(s, s, word, word, weight=calculate_arc_weight(dictionary[word])))
    print(s)


if __name__ == "__main__":
    words_syms = '../vocab/opensubtitles.syms'
    words_frec = '../vocab/opensubtitles.txt'

    make_W_fst(words_syms, words_frec)
