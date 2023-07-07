#!/bin/bash

#Declare an array of the state of the art CNN models
models=(SqueezeNet-v1.1 DenseNet-121 DenseNet-161 DenseNet-169 DenseNet-201 DenseNet-264 DPN-92 DPN-98 DPN-107 DPN-137 Inception-v1 Inception-v3 Xception InceptionResNetV2 MobileNet0.25-v1 MobileNet0.5-v1 MobileNet0.75-v1 MobileNet1.0-v1 MobileNet0.35-v2 MobileNet0.5-v2 MobileNet0.75-v2 MobileNet1.0-v2 MobileNet1.3-v2 MobileNet1.4-v2 MobileNet-small0.75-v3 MobileNet-small1.0-v3 MobileNet-small1.5-v3 MobileNet-large1.0-v3 ResNet-18 ResNet-34 ResNet-50 ResNet-101 ResNet-152 ResNet-50V2 ResNet-101V2 ResNet-152V2 ResNet-20 ResNet-32 ResNet-44 ResNet-56 ResNet-110 ResNet-164 ResNet-200 ResNet-20V2 ResNet-38V2 ResNet-47V2 ResNet-56V2 ResNet-110V2 ResNet-164V2 ResNext-50 ResNext-101 SEResNet-18 SEResNet-34 SEResNet-50 SEResNet-101 SEResNet-152 SEResNext-50 SEResNext-101 ShuffleNet0.5-v1 ShuffleNet1.0-v1 ShuffleNet1.5-v1 ShuffleNet2.0-v1 ShuffleNet0.5-v2 ShuffleNet1.0-v2 ShuffleNet1.5-v2 ShuffleNet2.0-v2 MNASNet0.35 MNASNet0.5 MNASNet0.75 MNASNet1.0 MNASNet1.4 EfficientNet-B0 EfficientNet-B1 EfficientNet-B2 EfficientNet-B3 EfficientNet-B4 SENet-154)

#Declare an array of the input sizes
image_sizes=(2400 1792 1600 1200 1024 896 800 720 600 568 512 480 448 331 320 299 256 240 224 150 128 112 90 75 64 56 32)

#Specify the batch size
batch_size=1

for model_ in ${models[@]}
do 
	for image_size in ${image_sizes[@]}
	do
		  echo "profiling "$model_"_.sqlite" ;
		  sudo /usr/local/cuda-10.0/bin/nvprof -o $model_"_"$image_size".sqlite" python3 Main.py $model_ $image_size $batch_size $1 ;
                  sleep 10;
	done
done

#Exit the script with a successful signal
exit 0
