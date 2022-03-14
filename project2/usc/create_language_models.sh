source ./path.sh

# constants
CURRENT_DIRECTORY=$(dirname $0)
# model files
LM_TRAIN=${CURRENT_DIRECTORY}/data/local/dict/lm_train.text
LM_DEV=${CURRENT_DIRECTORY}/data/local/dict/lm_dev.text
LM_TEST=${CURRENT_DIRECTORY}/data/local/dict/lm_test.text 
# temp models
UNIGRAM_TMP=${CURRENT_DIRECTORY}/data/local/lm_tmp/unigram_tmp.ilm.gz
BIGRAM_TMP=${CURRENT_DIRECTORY}/data/local/lm_tmp/bigram_tmp.ilm.gz
# compiled models
UNIGRAM_CMP=${CURRENT_DIRECTORY}/data/local/nist_lm/lm_phone_ug.arpa.gz
BIGRAM_CMP=${CURRENT_DIRECTORY}/data/local/nist_lm/lm_phone_bg.arpa.gz

# build models
build-lm.sh -i ${LM_TRAIN} -n 1 -o ${UNIGRAM_TMP}
build-lm.sh -i ${LM_TRAIN} -n 2 -o ${BIGRAM_TMP}

# compile models 
compile-lm ${UNIGRAM_TMP} -t=yes /dev/stdout | grep -v unk | gzip -c > ${UNIGRAM_CMP}
compile-lm ${BIGRAM_TMP} -t=yes /dev/stdout | grep -v unk | gzip -c > ${BIGRAM_CMP}

# create language fst (L.fst)
prepare_lang.sh data/local/dict "<oov>" data/local/lang data/lang

#create spk2utt files
utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

# create grammar fst (G.fst)
timit_format_data.sh

# calculate perplexity on dev and test
echo "--------------------- Perplexity of unigram model on dev ---------------------"
compile-lm ${UNIGRAM_TMP} -eval=${LM_DEV}
echo "--------------------- Perplexity of unigram model on test ---------------------"
compile-lm ${UNIGRAM_TMP} -eval=${LM_TEST}
echo "--------------------- Perplexity of bigram model on dev ---------------------"
compile-lm ${BIGRAM_TMP} -eval=${LM_DEV}
echo "--------------------- Perplexity of bigram model on test ---------------------"
compile-lm ${BIGRAM_TMP} -eval=${LM_TEST}

