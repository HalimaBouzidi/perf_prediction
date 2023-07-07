#!/bin/bash

#Declare an array of the indexs of the synthetic CNNs (the index in the generated_models.csv file which contains the description of the  synthetic CNN architectures)

models=(0 41 54 69 122 193 256 289 308 333 412 515 528 555 638 729 806 877 902 999 1076 1127 1164 1191 1234 1269 1298 1319 1398 1427 1476 1563 1606 1681 1746)


for model_ in ${models[@]}
do 
		 echo " Model : $model_ "
		 echo "Start of test"
		 python3 FLOPs_synth_calculator.py $model_ 
		 echo "End of test"
done

#Exit the script with a succefull signal
exit 0
