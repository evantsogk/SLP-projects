import numpy as np
from util import EPS, CHARS, format_arc, calculate_arc_weight


def edit_frequencies_dict(fname):
    """Create a dictionary with the edits and their frequency using add-1 smoothing.
    """
    # read edits
    with open(fname, 'r') as file:
        edits = ' '.join(file.read().splitlines()).split(' ')

    # create dictionary with frequencies
    unique_edits, counts = np.unique(edits, return_counts=True)
    dictionary = {}
    for i, edit in enumerate(unique_edits):
        edit = tuple(edit.split("\t"))
        dictionary[edit] = counts[i]

    # add-1 smoothing for edits that didn't appear
    for c in CHARS:
        if (c, EPS) not in dictionary:
            dictionary[(c, EPS)] = 0.5
        else:
            dictionary[(c, EPS)] += 0.5
        if (EPS, c) not in dictionary:
            dictionary[(EPS, c)] = 0.5
        else:
            dictionary[(EPS, c)] +=0.5
        for c_other in CHARS:
            if c_other != c:
                if (c, c_other) not in dictionary:
                    dictionary[(c, c_other)] = 0.5
                else:
                    dictionary[(c, c_other)] += 0.5

    total_appearances = sum(counts)
    # store probabilities
    for edit, freq in dictionary.items():
        dictionary[edit] = freq / total_appearances

    return dictionary


def make_E_fst(edit_freq):
    """
    Create a transducer like Levenshtein whose arc weights are the negative log frequency of the edits. Add-1 smoothing
    is used for edits that didn't appear in the wiki corpus.
    """
    s = 0  # the only state

    for c in CHARS:
        # no edit (cost 0)
        print(format_arc(s, s, c, c, weight=0))
        # deletion
        print(format_arc(s, s, c, EPS, weight=calculate_arc_weight(edit_freq[(c, EPS)])))
        # insertion
        print(format_arc(s, s, EPS, c, weight=calculate_arc_weight(edit_freq[(EPS, c)])))
        # substitution
        for c_other in CHARS:
            if c_other != c:
                print(format_arc(s, s, c, c_other, weight=calculate_arc_weight(edit_freq[(c, c_other)])))
    print(s)


if __name__ == "__main__":
    edits_path = '../vocab/edits.txt'

    edits_freq = edit_frequencies_dict(edits_path)
    make_E_fst(edits_freq)
