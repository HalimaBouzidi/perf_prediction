#!/bin/bash

for value in {1..50}
do
	echo $value
	python3 Model_synth_gen.py 1
done

#Exit the script with a successful signal
exit 0
