import re


def create_wordsyms(fname_in, fname_out):
    """
    Create word.syms from given dictionary
    """
    def isEnglish(s):
        """
        Return true if string contains only english characters
        """
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    with open(fname_in, 'r', encoding="utf8") as file:
        raw = file.read()
    raw = raw.strip()  # strip leading / trailing spaces
    raw = raw.lower()  # convert to lowercase
    raw = re.sub("\s+", " ", raw)  # strip multiple whitespace
    tokens = [token for token in raw.split(' ') if token.isalpha() and isEnglish(token)]  # ignore tokens that contain non english letters

    # write word.syms file
    with open(fname_out, 'w', encoding="utf8") as file:
        symbol = 0
        file.write('<eps>' + '\t' + str(symbol) + '\n')
        for token in sorted(tokens):
            symbol += 1
            file.write(token + '\t' + str(symbol) + '\n')


if __name__ == "__main__":
    simpsons_txt = '../vocab/simpsons.txt'
    simpsons_syms = '../vocab/simpsons.syms'
    opensubtitles_txt = '../vocab/opensubtitles.txt'
    opensubtitles_syms = '../vocab/opensubtitles.syms'

    # create wordsyms files from the Simpsons and the opensubtitles word frequency lists
    create_wordsyms(simpsons_txt, simpsons_syms)
    create_wordsyms(opensubtitles_txt, opensubtitles_syms)
