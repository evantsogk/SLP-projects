source ./path.sh

# train monophone GMM-HMM
steps/train_mono.sh --nj 4 data/train data/lang exp/mono

# create HCLG graphs for unigram and bigram models
utils/mkgraph.sh data/lang_phones_ug exp/mono exp/mono/graph_ug
utils/mkgraph.sh data/lang_phones_bg exp/mono exp/mono/graph_bg

# decode sentences of validation and test set
steps/decode.sh --nj 4 exp/mono/graph_ug data/dev exp/mono/decode_dev_ug
steps/decode.sh --nj 4 exp/mono/graph_ug data/test exp/mono/decode_test_ug
steps/decode.sh --nj 4 exp/mono/graph_bg data/dev exp/mono/decode_dev_bg
steps/decode.sh --nj 4 exp/mono/graph_bg data/test exp/mono/decode_test_bg

# print PER
cat exp/mono/decode_dev_ug/scoring_kaldi/best_wer
cat exp/mono/decode_test_ug/scoring_kaldi/best_wer
cat exp/mono/decode_dev_bg/scoring_kaldi/best_wer
cat exp/mono/decode_test_bg/scoring_kaldi/best_wer

# align phonems using monophone model
steps/align_si.sh --nj 4 data/train data/lang exp/mono exp/mono_ali

# train triphone model using the monophone alignments
steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_ali exp/tri

# create HCLG graphs for unigram and bigram models
utils/mkgraph.sh data/lang_phones_ug exp/tri exp/tri/graph_ug
utils/mkgraph.sh data/lang_phones_bg exp/tri exp/tri/graph_bg

# decode sentences of validation and test set
steps/decode.sh --nj 4 exp/tri/graph_ug data/dev exp/tri/decode_dev_ug
steps/decode.sh --nj 4 exp/tri/graph_ug data/test exp/tri/decode_test_ug
steps/decode.sh --nj 4 exp/tri/graph_bg data/dev exp/tri/decode_dev_bg
steps/decode.sh --nj 4 exp/tri/graph_bg data/test exp/tri/decode_test_bg

# print PER
cat exp/tri/decode_dev_ug/scoring_kaldi/best_wer
cat exp/tri/decode_test_ug/scoring_kaldi/best_wer
cat exp/tri/decode_dev_bg/scoring_kaldi/best_wer
cat exp/tri/decode_test_bg/scoring_kaldi/best_wer


