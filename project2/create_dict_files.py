# create files in data/local/dict

from pathlib import Path
import numpy as np


def create_lm_file(input_file, output_file):
    with open(input_file) as f:
        texts = f.read().splitlines()
    with open(output_file, 'w+') as f:
        for text in texts:
            f.write('<s> ' + text.split(' ', 1)[1] + ' </s>\n')


if __name__ == "__main__":
    dict_dir = Path(Path.home(), 'kaldi/egs/usc/data/local/dict')
    dict_dir.mkdir(parents=True, exist_ok=True)

    # create silence_phones.txt and optional_silence.txt
    with open(Path(dict_dir, 'silence_phones.txt'), 'w+') as f:
        f.write('sil\n')
    with open(Path(dict_dir, 'optional_silence.txt'), 'w+') as f:
        f.write('sil\n')

    # create an array with unique phones sorted
    phones = []
    with open('slp_lab2_data/lexicon.txt') as f:
        for line in f:
            phones.extend(line[:-1].split(' ')[1:])
    phones.remove('')
    phones.remove('sil')
    phones = np.unique(phones)

    # create nonsilence_phones.txt
    with open(Path(dict_dir, 'nonsilence_phones.txt'), 'w+') as f:
        for phone in phones:
            f.write(phone + '\n')

    # create lexicon.txt
    with open(Path(dict_dir, 'lexicon.txt'), 'w+') as f:
        f.write('sil sil\n')
        for phone in phones:
            f.write(phone + ' ' + phone + '\n')

    # create lm_train.text, lm__dev.text, lm_test.text
    create_lm_file(Path(Path.home(), 'kaldi/egs/usc/data/train/text'), Path(dict_dir, 'lm_train.text'))
    create_lm_file(Path(Path.home(), 'kaldi/egs/usc/data/dev/text'), Path(dict_dir, 'lm_dev.text'))
    create_lm_file(Path(Path.home(), 'kaldi/egs/usc/data/test/text'), Path(dict_dir, 'lm_test.text'))

    # create extra_questions.txt
    with open(Path(dict_dir, 'extra_questions.txt'), 'w+') as f:
        pass
