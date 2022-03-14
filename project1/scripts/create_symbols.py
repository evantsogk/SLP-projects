def map_chars():
    """ Creates a dictionary that maps each lower case character to a unique index."""
    dictionary = {'<eps>': 0}
    for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
        dictionary[char] = i + 1
    return dictionary


def map_words(tokens):
    """ Creates a dictionary that maps each word to a unique index."""
    dictionary = {'<eps>': 0}
    for i, token in enumerate(tokens):
        dictionary[token] = i + 1
    return dictionary


def save_dict(fname, dictionary):
    """ Save a dictionary to a file with tab separated columns."""
    with open(fname, 'w') as file:
        for key, value in dictionary.items():
            file.write(key + '\t' + str(value) + '\n')


if __name__ == "__main__":
    char_sym_path = '../vocab/chars.syms'
    word_sym_path = '../vocab/words.syms'
    tokens_path = '../vocab/words.vocab.txt'

    # create character to index file
    char_dict = map_chars()
    save_dict(char_sym_path, char_dict)

    # read tokens
    with open(tokens_path, 'r') as file:
        tokens = [line.split('\t')[0] for line in file.readlines()]
    # create word to index file
    word_dict = map_words(tokens)
    save_dict(word_sym_path, word_dict)
