CURRENT_DIRECTORY=$(dirname $0)
SPELLCHECKER=${CURRENT_DIRECTORY}/../fsts/S.binfst

for WORD in contenpted contende contended contentid begining problam proble promblem proplen dirven exstacy ecstacy guic juce jucie juise juse localy compair pronounciation
do
	bash predict.sh ${SPELLCHECKER} $WORD
	printf "\n"
done

