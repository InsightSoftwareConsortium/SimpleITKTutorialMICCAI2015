#!/usr/bin/env bash

thisdir="${BASH_SOURCE%/*}"

tempoutput=`tempfile --suffix=.ipynb`
trap 'rm -f $MYTEMP' SIGINT SIGTERM

failed=0
while read notebook; do
  echo $notebook | grep -q ipynb_checkpoints && continue
  echo "Testing ${notebook}..."
  jupyter nbconvert --to notebook "$notebook" \
    --output $tempoutput \
    --execute \
    --log-level=DEBUG \
    --ExecutePreprocessor.timeout=300
  if test $? -ne 0; then
    echo "Error: $(basename "$notebook") failed to execute." 1>&2
    let "failed++"
  fi
done <<< "$(find "$thisdir" -name '*.ipynb')"

rm $tempoutput
echo "$failed notebooks failed to execute."
exit $failed
