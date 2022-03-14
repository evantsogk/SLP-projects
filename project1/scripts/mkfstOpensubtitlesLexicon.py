from util import EPS, format_arc


def make_lexicon_fst(words):
    """Create an fsa that accepts the words of the vocabulary
    """
    s = 0  # s=0 is the one and only initial state
    for word in words:
        for i in range(0, len(word)):
            if i == 0:
                # from the first letter we go to the word starting from the initial state
                print(format_arc(0, s + 1, word[i], word, weight=0))
            else:
                s += 1
                print(format_arc(s, s + 1, word[i], EPS, weight=0))  # the rest of the letters are epsilon transitions
        s += 1
        print(s)  # accept state at the end of each word


if __name__ == "__main__":
    tokens_path = '../vocab/opensubtitles.syms'

    # read tokens
    with open(tokens_path, 'r') as file:
        tokens = [line.split('\t')[0] for line in file.readlines()]
        tokens.remove(EPS)

    make_lexicon_fst(tokens)

