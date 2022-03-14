import glob
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors

SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(SCRIPT_DIRECTORY, "../data/aclImdb/")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = -1
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = None


SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)


class W2VLossLogger(CallbackAny2Vec):
    """Callback to print loss after each epoch
    use by passing model.train(..., callbacks=[W2VLossLogger()])
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()

        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print(
                "Loss after epoch {}: {}".format(
                    self.epoch, loss - self.loss_previous_step
                )
            )
        self.epoch += 1
        self.loss_previous_step = loss


def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)


def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())


def tokenize(s):
    return s.split(" ")


def preproc_tok(s):
    return tokenize(preprocess(s))


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []

    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, "r", encoding="utf-8") as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)

    return data


def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])


def extract_nbow(corpus, word_vectors):
    """Extract neural bag of words representations"""
    nbow = []
    for sentence in corpus:
        sentence_vector = np.mean([word_vectors[word] if word in word_vectors else word_vectors.vector_size*[0] for word in sentence.split(" ")], axis=0)
        nbow.append(sentence_vector)
    return np.asarray(nbow)


def train_sentiment_analysis(train_corpus, train_labels):
    """Train a sentiment analysis classifier using NBOW + Logistic regression"""
    clf = LogisticRegression()
    clf.fit(train_corpus, train_labels)
    return clf


def evaluate_sentiment_analysis(classifier, test_corpus, test_labels):
    """Evaluate classifier in the test corpus and report accuracy"""
    accuracy = classifier.score(test_corpus, test_labels)
    print("Accuracy =", accuracy)


if __name__ == "__main__":
    # read Imdb corpus
    pos_data = read_samples(pos_train_dir)
    pos_data.extend(read_samples(pos_test_dir))
    neg_data = read_samples(neg_train_dir)
    neg_data.extend(read_samples(neg_test_dir))

    corpus, labels = create_corpus(pos_data, neg_data)

    # load models
    model_gutenberg = Word2Vec.load("gutenberg_w2v.100d.model").wv
    model_google = KeyedVectors.load_word2vec_format(
        os.path.join(SCRIPT_DIRECTORY, '../data/GoogleNews-vectors-negative300.bin'), binary=True, limit=NUM_W2V_TO_LOAD)

    for model in [model_gutenberg, model_google]:
        # create nbow
        nbow_corpus = extract_nbow(corpus, model)
        train_corpus, test_corpus, train_labels, test_labels = train_test_split(nbow_corpus, labels)

        # train and evaluate
        trained_clf = train_sentiment_analysis(train_corpus, train_labels)
        evaluate_sentiment_analysis(trained_clf, test_corpus, test_labels)
