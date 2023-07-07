#!/bin/bash

#in another terminal, I'll execute the following script, in order to monitor the memory usage

while true; do 
	echo -e "`date`\n\n`free`" >> free.txt ; 
	sleep 2; 
done

exit 0
