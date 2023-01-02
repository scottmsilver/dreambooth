#!/bin/bash

for booth in 1 2 3 4 5
do
  python dream.py build-model --username=scott --booth-id=$booth
  python dream.py generate-images --username=scott --booth-id=$booth  --prompt-file=prompts.txt
done
