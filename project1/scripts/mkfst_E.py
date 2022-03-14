import numpy as np
from util import EPS, CHARS, INFINITY, format_arc, calculate_arc_weight


def edit_frequencies_dict(fname):
    # read edits
    with open(fname, 'r') as file:
        edits = ' '.join(file.read().splitlines()).split(' ')

    # create dictionary with frequencies
    unique_edits, counts = np.unique(edits, return_counts=True)
    dictionary = {}
    for i, edit in enumerate(unique_edits):
        edit = tuple(edit.split("\t"))
        dictionary[edit] = counts[i] / np.sum(counts)

    return dictionary


def make_E_fst(edit_freq):
    """
    Create a transducer like Levenshtein whose arc weights are the negative log frequency of the edits.
    """
    s = 0  # the only state

    for c in CHARS:
        # no edit (cost 0)
        print(format_arc(s, s, c, c, weight=0))
        # deletion
        if (c, EPS) in edit_freq:
            print(format_arc(s, s, c, EPS, weight=calculate_arc_weight(edit_freq[(c, EPS)])))
        else:
            print(format_arc(s, s, c, EPS, weight=INFINITY))
        # insertion
        if (EPS, c) in edit_freq:
            print(format_arc(s, s, EPS, c, weight=calculate_arc_weight(edit_freq[(EPS, c)])))
        else:
            print(format_arc(s, s, EPS, c, weight=INFINITY))
        # substitution
        for c_other in CHARS:
            if c_other != c:
                if (c, c_other) in edit_freq:
                    print(format_arc(s, s, c, c_other, weight=calculate_arc_weight(edit_freq[(c, c_other)])))
                else:
                    print(format_arc(s, s, c, c_other, weight=INFINITY))
    print(s)


if __name__ == "__main__":
    edits_path = '../vocab/edits.txt'

    edits_freq = edit_frequencies_dict(edits_path)
    make_E_fst(edits_freq)
