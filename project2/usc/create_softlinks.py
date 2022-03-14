# create soft links

from pathlib import Path
import os

if __name__ == "__main__":
	# steps soft link
	src = Path(Path.home(), 'kaldi/egs/wsj/s5/steps')
	dst = Path(Path.home(), 'kaldi/egs/usc/steps')
	if not os.path.islink(dst):
		os.symlink(src, dst, target_is_directory = True)

# utils soft link
	src = Path(Path.home(), 'kaldi/egs/wsj/s5/utils')
	dst = Path(Path.home(), 'kaldi/egs/usc/utils')
	if not os.path.islink(dst):
		os.symlink(src, dst, target_is_directory = True)

	# create directory local
	local_dir = Path(Path.home(), 'kaldi/egs/usc/local')
	local_dir.mkdir(parents=True, exist_ok=True)

	# score_kaldi.sh soft link
	src = Path(Path.home(), 'kaldi/egs/wsj/s5/steps/score_kaldi.sh')
	dst = Path(Path.home(), 'kaldi/egs/usc/local/score_kaldi.sh')
	if not os.path.islink(dst):
		os.symlink(src, dst)

	# create other directories
	Path(Path.home(), 'kaldi/egs/usc/conf').mkdir(parents=True, exist_ok=True)
	Path(Path.home(), 'kaldi/egs/usc/data/lang').mkdir(parents=True, exist_ok=True)
	Path(Path.home(), 'kaldi/egs/usc/data/local/dict').mkdir(parents=True, exist_ok=True)
	Path(Path.home(), 'kaldi/egs/usc/data/local/lm_tmp').mkdir(parents=True, exist_ok=True)
	Path(Path.home(), 'kaldi/egs/usc/data/local/nist_lm').mkdir(parents=True, exist_ok=True)
	

