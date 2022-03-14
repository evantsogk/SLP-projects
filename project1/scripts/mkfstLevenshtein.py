from util import EPS, CHARS, format_arc


def make_levenshtein_fst():
    """Create an fst with one state that implements the Levenshtein distance
    """
    s = 0  # the only state

    for c in CHARS:
        # no edit (cost 0)
        print(format_arc(s, s, c, c, weight=0))
        # deletion (cost 1)
        print(format_arc(s, s, c, EPS, weight=1))
        # insertion (cost 1)
        print(format_arc(s, s, EPS, c, weight=1))
        # substitution (cost 1)
        for c_other in CHARS:
            if c_other != c:
                print(format_arc(s, s, c, c_other, weight=1))
    print(s)


if __name__ == "__main__":
    make_levenshtein_fst()
