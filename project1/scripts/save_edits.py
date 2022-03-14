import os
from tqdm import tqdm
from helpers import run_cmd, read_wiki_txt


def word_edits(wrong_word, correct_word):
    # run the script word_edits.sh for a pair of (wrong, correct) words to find the edits needed for the correction
    edits = run_cmd(f"bash word_edits.sh {wrong_word} {correct_word}")

    return edits


if __name__ == "__main__":
    wiki_text_path = '../data/wiki.txt'
    edits_path = '../vocab/edits.txt'

    # read the pairs of wrong and correct words
    wiki_pairs = read_wiki_txt(wiki_text_path)

    # find word edits for each pair of wrong and correct word
    edits = ''
    for pair in wiki_pairs:
        tqdm.write(f"{pair[0]} -> {pair[1]}")
        edits += word_edits(*pair)

    # save edits to file
    with open(edits_path, "w") as text_file:
        text_file.write(edits)
