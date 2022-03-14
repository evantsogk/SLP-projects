#!/usr/bin/env bash
source ./path.sh

DATA_PATH=./data/test

GRAPH_PATH=./exp/tri/graph_bg
TEST_ALI_PATH=./exp/tri_ali_test
OUT_DECODE_PATH=./exp/tri/decode_test_dnn

CHECKPOINT_FILE=./best_usc_dnn.pt
DNN_OUT_FOLDER=./dnn_out

# ------------------- Data preparation for DNN -------------------- #
# extract triphone alignments for train validation and test sets
steps/align_si.sh --nj 4 data/train data/lang exp/tri exp/tri_ali_train
steps/align_si.sh --nj 4 data/dev data/lang exp/tri exp/tri_ali_dev
steps/align_si.sh --nj 4 data/test data/lang exp/tri exp/tri_ali_test

# Compute cmvn stats for every set and save them in specific .ark files
# These will be used by the python dataset class that you were given
for set in train dev test; do
  compute-cmvn-stats --spk2utt=ark:data/${set}/spk2utt scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_speaker.ark"
  compute-cmvn-stats scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_snt.ark"
done

# ------------------ TRAIN DNN ------------------------------------ #
python3 timit_dnn.py $CHECKPOINT_FILE


# ----------------- EXTRACT DNN POSTERIORS ------------------------ #
python3 extract_posteriors.py $CHECKPOINT_FILE $DNN_OUT_FOLDER


# ----------------- RUN DNN DECODING ------------------------------ #
./decode_dnn.sh $GRAPH_PATH $DATA_PATH $TEST_ALI_PATH $OUT_DECODE_PATH "cat $DNN_OUT_FOLDER/posteriors.ark"

