source ./path.sh

# extract mfccs
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 4 data/train
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 4 data/dev
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 4 data/test

# normalisation
steps/compute_cmvn_stats.sh data/train
steps/compute_cmvn_stats.sh data/dev
steps/compute_cmvn_stats.sh data/test

# number of frames of first 5 sentences
feat-to-len scp:data/train/feats.scp ark,t:data/train/feats.lengths
head -n 5 data/train/feats.lengths

# feature dimension
feat-to-dim scp:data/train/feats.scp -

