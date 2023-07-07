#!/bin/bash

#Declare an array of the synthetic CNNs
models=(basic_model_750_19 basic_model_75_5 basic_model_56_6 basic_model_224_25 basic_model_350_34 basic_model_500_30 basic_model_620_15 basic_model_350_8 basic_model_112_11 basic_model_720_38 basic_model_200_50 basic_model_720_5 basic_model_64_12 basic_model_512_40 basic_model_224_44 basic_model_224_37 basic_model_150_34 basic_model_850_11 basic_model_56_47 basic_model_720_37 basic_model_820_24 basic_model_112_17 basic_model_800_12 basic_model_700_20 basic_model_850_16 basic_model_256_13 basic_model_90_9 basic_model_224_38 basic_model_331_13 basic_model_320_23 basic_model_600_42 basic_model_320_20 basic_model_600_36 basic_model_600_31 basic_model_75_12)

for model_ in ${models[@]}
do 
		  echo " Model : $model_ with image size : $image_size"
		  echo "Start of test"
		  python3 Parser_reader_synth.py $model_ $image_size
		  echo "End of test"
done

#Exit the script with a succefull signal
exit 0
