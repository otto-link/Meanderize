#!/bin/bash

# python
DIRS="."

for i in ${DIRS}; do
    yapf -vv --in-place --recursive "$i/."
done

# clean-up
find . -type f -name 'yapf*.py' -exec rm -f {} +
