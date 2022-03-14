import os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors

SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
NUM_W2V_TO_LOAD = 500000


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


if __name__ == "__main__":
    # load model trained on gutenberg
    model_gutenberg = Word2Vec.load("gutenberg_w2v.100d.model")
    model_google = KeyedVectors.load_word2vec_format(os.path.join(SCRIPT_DIRECTORY, '../data/GoogleNews-vectors-negative300.bin'),
                                                     binary=True, limit=NUM_W2V_TO_LOAD)

    # STEP 12: Word representations
    # similar words
    print("Gutenberg model...")
    for word in ['bible', 'book', 'bank', 'water']:
        similar = model_gutenberg.wv.most_similar(positive=[word])
        print('Similar words to "' + word + '":', similar)

    print("Google News model...")
    for word in ['bible', 'book', 'bank', 'water']:
        similar = model_google.most_similar(positive=[word])
        print('Similar words to "' + word + '":', similar)

    # analogies
    print("Gutenberg model...")
    for triplet in [('girls', 'queens', 'kings'), ('good', 'taller', 'tall'), ('france', 'paris', 'london')]:
        result = model_gutenberg.wv.most_similar(positive=[triplet[0], triplet[2]], negative=[triplet[1]])
        print(triplet[0] + ' - ' + triplet[1] + ' + ' + triplet[2] + ' =', result[0])

    print("Google News model...")
    for triplet in [('girls', 'queens', 'kings'), ('good', 'taller', 'tall'), ('france', 'paris', 'london')]:
        result = model_google.most_similar(positive=[triplet[0], triplet[2]], negative=[triplet[1]])
        print(triplet[0] + ' - ' + triplet[1] + ' + ' + triplet[2] + ' =', result[0])

    # STEP 13:
    # save trained word embeddings
    with open('embeddings.tsv', 'w') as embeddings:
        with open('metadata.tsv', 'w') as metadata:
            for word in model_gutenberg.wv.vocab.keys():
                embeddings.write('\t'.join(map(str, model_gutenberg[word])) + '\n')
                metadata.write(word + '\n')
