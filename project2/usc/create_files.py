# create all initial files: uttids.txt, utt2spk.txt, wav.scp, text.txt

from pathlib import Path
import re


def create_files(output_dir, fileset):
    with open('slp_lab2_data/filesets/' + fileset) as f:
        uttids = f.read().splitlines()
    spkids = [utid.split('_')[2] for utid in uttids]
    wavs = [str(Path(Path().absolute(), 'slp_lab2_data/wav', spkids[i], uttids[i] + '.wav')) for i in range(len(uttids))]

    # data for text file
    with open('slp_lab2_data/transcription.txt') as f:
        transcripts = f.read().splitlines()
    transcripts = [re.sub('[^a-zA-Z \']+', ' ', transcript.upper()) for transcript in transcripts]

    lexicon = {}
    with open('slp_lab2_data/lexicon.txt') as f:
        for line in f:
            (key, val) = re.split('\t| {2}', line[:-1])
            lexicon[key] = val
    transcripts = ['sil' + ''.join([lexicon[word] for word in sentence.split()]) + ' sil' for sentence in transcripts]

    texts = [transcripts[int(utid.split('_')[3])-1] for utid in uttids]

    # create uttids.txt
    with open(Path(output_dir, 'uttids'), 'w+') as f:
        for uttid in uttids:
            f.write(uttid + '\n')

    # create utt2spk.txt
    with open(Path(output_dir, 'utt2spk'), 'w+') as f:
        for uttid, spk in zip(uttids, spkids):
            f.write(uttid + ' ' + spk + '\n')

    # create wav.scp
    with open(Path(output_dir, 'wav.scp'), 'w+') as f:
        for uttid, wav in zip(uttids, wavs):
            f.write(uttid + ' ' + wav + '\n')

    # create text.txt
    with open(Path(output_dir, 'text'), 'w+') as f:
        for uttid, text in zip(uttids, texts):
            f.write(uttid + ' ' + text + '\n')


if __name__ == "__main__":
    train_dir = Path(Path.home(), 'kaldi/egs/usc/data/train')
    dev_dir = Path(Path.home(), 'kaldi/egs/usc/data/dev')
    test_dir = Path(Path.home(), 'kaldi/egs/usc/data/test')

    # create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # create files
    create_files(train_dir, 'train_utterances.txt')
    create_files(dev_dir, 'validation_utterances.txt')
    create_files(test_dir, 'test_utterances.txt')
