import argparse
import time
import csv
from statistics import mean
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K
import numpy as np
from classification_models.keras import Classifiers
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

# Pre-allocate a small portion of the GPU memory, and then grow the memory allocation grow as needed
# Nano_portion = 0.1, TX2_portion = 0.05 & AGX_portion = 0.025
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf_config.gpu_options.allow_growth = True
# Create session and load graph
tf_sess = tf.Session(config=tf_config)
K.set_session(tf_sess)

# the following function will transorm a given image to an array of the specified size (image_size as (H, W, C))
def image_to_tensor(image_path, image_size, model_name, not_batch) :
    img = image.load_img(image_path, target_size=image_size[:2])
    x = image.img_to_array(img)
    if (not_batch) : 
        x = np.expand_dims(x, axis=0)
    
    if(model_name == 'DenseNet-121' or model_name == 'DenseNet-161' or model_name == 'DenseNet-169' or model_name == 'DenseNet-201') :
        from keras.applications.densenet import preprocess_input as preprocess_input_densenet
        x = preprocess_input_densenet(x)
    elif(model_name == 'DenseNet-161' or model_name == 'DenseNet-264') :
        from keras_contrib.applications.densenet import preprocess_input as preprocess_input_densenet
        x = preprocess_input_densenet(x)
    
    elif (model_name == 'Inception-v1' or model_name == 'Inception-v3') :
        from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
        x = preprocess_input_inception_v3(x)
    elif (model_name == 'Xception') :
        from keras.applications.xception import preprocess_input as preprocess_input_xception
        x = preprocess_input_xception(x)
    elif (model_name == 'InceptionResNetV2') :
        from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
        x = preprocess_input_inception_resnet_v2(x)

    elif (model_name == 'MobileNet0.25-v1' or model_name == 'MobileNet0.5-v1'  or model_name == 'MobileNet0.75-v1' or model_name == 'MobileNet1.0-v1') :
        from keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet
        x = preprocess_input_mobilenet(x)
    elif (model_name == 'MobileNet0.35-v2' or model_name == 'MobileNet0.5-v2'  or model_name == 'MobileNet0.75-v2' or model_name == 'MobileNet1.0-v2' or model_name == 'MobileNet1.3-v2'or model_name == 'MobileNet1.4-v2') :
        from keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2
        x = preprocess_input_mobilenet_v2(x)

    elif (model_name == 'MobileNet-small0.75-v3' or model_name == 'MobileNet-small1.0-v3' or model_name == 'MobileNet-small1.5-v3' or model_name == 'MobileNet-large1.0-v3') :
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)
    
    elif (model_name == 'ResNet-18') :
        preprocess_input = Classifiers.get('resnet18')[1]
        x = preprocess_input(x)
    elif (model_name == 'ResNet-34') :
        preprocess_input = Classifiers.get('resnet34')[1]
        x = preprocess_input(x)
    elif (model_name == 'ResNet-50') :
        from keras.applications.resnet50 import preprocess_input as preprocess_input_resent50
        x = preprocess_input_resent50(x)
    elif (model_name == 'ResNet-101' or model_name == 'ResNet-152') :
        from keras.applications.resnet import preprocess_input as preprocess_input_resent_101_152
        x = preprocess_input_resent_101_152(x)

    elif (model_name == 'ResNet-50V2' or model_name == 'ResNet-101V2' or model_name == 'ResNet-152V2') :
        from keras.applications.resnet_v2 import preprocess_input as preprocess_input_resent_V2
        x = preprocess_input_resent_V2(x)

    elif (model_name == 'ResNext-50') :
        preprocess_input = Classifiers.get('resnext50')[1]
        x = preprocess_input(x)
    elif (model_name == 'ResNext-101') :
        preprocess_input = Classifiers.get('resnext101')[1]
        x = preprocess_input(x)

    elif (model_name == 'SEResNet-18') :
        preprocess_input = Classifiers.get('seresnet18')[1]
        x = preprocess_input(x)
    elif (model_name == 'SEResNet-34') :
        preprocess_input = Classifiers.get('seresnet34')[1]
        x = preprocess_input(x)
    elif (model_name == 'SEResNet-50') :
        preprocess_input = Classifiers.get('seresnet50')[1]
        x = preprocess_input(x)
    elif (model_name == 'SEResNet-101') :
        preprocess_input = Classifiers.get('seresnet101')[1]
        x = preprocess_input(x)
    elif (model_name == 'SEResNet-152') :
        preprocess_input = Classifiers.get('seresnet152')[1]
        x = preprocess_input(x)

    elif (model_name == 'SEResNext-50') :
        preprocess_input = Classifiers.get('seresnext50')[1]
        x = preprocess_input(x)
    elif (model_name == 'SEResNext-101') :
        preprocess_input = Classifiers.get('seresnext101')[1]
        x = preprocess_input(x)

    elif (model_name == 'SENet-154') :
        preprocess_input = Classifiers.get('senet154')[1]
        x = preprocess_input(x)

    elif (model_name == 'SqueezeNet-v1.1') :
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)

    elif (model_name == 'VGG-16') :
        from keras.applications.vgg16 import preprocess_input
        x = preprocess_input(x)
    elif (model_name == 'VGG-19') :
        from keras.applications.vgg19 import preprocess_input
        x = preprocess_input(x)

    elif (model_name == 'NASNet-Mobile' or model_name == 'NASNet-Large') :
        from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
        x = preprocess_input_nasnet(x)

    elif (model_name == 'EfficientNet-B0' or model_name == 'EfficientNet-B1' or model_name == 'EfficientNet-B2' or model_name == 'EfficientNet-B3' or model_name == 'EfficientNet-B4' or model_name == 'EfficientNet-B5' or model_name == 'EfficientNet-B6' or model_name == 'EfficientNet-B7') :
        from efficientnet.tfkeras import preprocess_input as preprocess_input_efficientnet
        x = preprocess_input_efficientnet(x)

    elif (model_name == 'MNASNet0.35' or model_name == 'MNASNet0.5' or model_name == 'MNASNet0.75' or model_name == 'MNASNet1.0' or model_name == 'MNASNet1.4') :
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)

    elif (model_name == 'DPN-92' or model_name == 'DPN-98' or model_name == 'DPN-107' or model_name == 'DPN-137') :
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)

    elif (model_name == 'ShuffleNet0.5-v1' or model_name == 'ShuffleNet1.0-v1' or model_name == 'ShuffleNet1.5-v1' or model_name == 'ShuffleNet2.0-v1') :
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)

    elif (model_name == 'ShuffleNet0.5-v2' or model_name == 'ShuffleNet1.0-v2' or model_name == 'ShuffleNet1.5-v2' or model_name == 'ShuffleNet2.0-v2') :
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)
    
    elif (model_name == 'ResNet-20' or model_name == 'ResNet-32' or model_name == 'ResNet-44' or model_name == 'ResNet-56' or model_name == 'ResNet-110' or model_name == 'ResNet-164') :
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)

    elif (model_name == 'ResNet-20V2' or model_name == 'ResNet-38V2' or model_name == 'ResNet-47V2' or model_name == 'ResNet-56V2' or model_name == 'ResNet-110V2' or model_name == 'ResNet-164V2' or model_name == 'ResNet-1001V2') :
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)

    return x

# The following function deserialize the frozen graph in order to import it to the tensorflow graph, of the actual session.
def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk. (from the rep Saved-Model)"""
    with TF_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
    return graph_def

# The followin function will create a csv file, which contains the times spent on the predections, recorded for the 30 iterastions
def create_csv(model_name, batch_size, times, output_path):
    with open(output_path, 'w', newline='') as file :
        writer = csv.writer(file)
        writer.writerow([""])
        writer.writerow(["Model name : "+model_name, "Batch_size : "+batch_size])
        writer.writerow(["iteration", "inference time per batch", "inference time per image"])
        for idx, time in enumerate(times, start=0) :
            writer.writerow([idx+1, time*1000, (time/int(batch_size))*1000])


	#************************************ Start of the main script ******************************************* #

if __name__ == "__main__" :

    with tf_sess :

        print('gpu dispo : ', tf.test.is_gpu_available())

        parser = argparse.ArgumentParser()
        parser.add_argument("model_name", default="MobileNet-v1", type=str, help="Specifiy the name of the model")
        parser.add_argument("image_size", default="75", type=str, help="Specifiy the image size")
        parser.add_argument("batch_size", default="1", type=str, help="Specifiy the batch size")
        parser.add_argument("target_device", default="Nano", type=str, help="Specifiy the target device")
        args = parser.parse_args()

        models_infos = { 
        "DenseNet-121" : {"path": "./Models/DenseNet-121","tf_graph" : "DenseNet-121-tf-graph",
            "output_names" : "fc1000/Softmax", "input_names" : "input_1"},

        "DenseNet-161" : {"path": "./Models/DenseNet-161","tf_graph" : "DenseNet-161-tf-graph",
            "output_names" : "DenseNet/dense_1/truediv", "input_names" : "input_1"},

        "DenseNet-169" : {"path": "./Models/DenseNet-169","tf_graph" : "DenseNet-169-tf-graph",
            "output_names" : "fc1000/Softmax", "input_names" : "input_1"},

        "DenseNet-201" : {"path": "./Models/DenseNet-201","tf_graph" : "DenseNet-201-tf-graph",
            "output_names" : "fc1000/Softmax", "input_names" : "input_1"},

        "DenseNet-264" : {"path": "./Models/DenseNet-264","tf_graph" : "DenseNet-264-tf-graph",
            "output_names" : "DenseNet/dense_1/truediv", "input_names" : "input_1"},

        "DPN-92" : {"path": "./Models/DPN-92","tf_graph" : "DPN-92-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "DPN-98" : {"path": "./Models/DPN-98","tf_graph" : "DPN-98-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "DPN-107" : {"path": "./Models/DPN-107","tf_graph" : "DPN-107-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "DPN-137" : {"path": "./Models/DPN-137","tf_graph" : "DPN-137-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "Inception-v1" : {"path" : "./Models/Inception-v1", "tf_graph" : "Inception-v1-tf-graph",
            "output_names" : "Predictions/Softmax", "input_names" : "input_1"},

        "Inception-v3" : {"path" : "./Models/Inception-v3", "tf_graph" : "Inception-v3-tf-graph",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},

        "InceptionResNetV2" : {"path" : "./Models/InceptionResNetV2", "tf_graph" : "InceptionResNetV2-tf-graph",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},

        "Xception" : {"path" : "./Models/Xception", "tf_graph" : "Xception-tf-graph",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},

        "MobileNet0.25-v1" : {"path" : "./Models/MobileNet0.25-v1", "tf_graph" : "MobileNet0.25-v1-tf-graph",
            "output_names" : "act_softmax/Softmax", "input_names" : "input_1"},

        "MobileNet0.5-v1" : {"path" : "./Models/MobileNet0.5-v1", "tf_graph" : "MobileNet0.5-v1-tf-graph",
            "output_names" : "act_softmax/Softmax", "input_names" : "input_1"},

        "MobileNet0.75-v1" : {"path" : "./Models/MobileNet0.75-v1", "tf_graph" : "MobileNet0.75-v1-tf-graph",
            "output_names" : "act_softmax/Softmax", "input_names" : "input_1"},

        "MobileNet1.0-v1" : {"path" : "./Models/MobileNet1.0-v1", "tf_graph" : "MobileNet1.0-v1-tf-graph",
            "output_names" : "act_softmax/Softmax", "input_names" : "input_1"},

        "MobileNet0.35-v2" :  {"path" : "./Models/MobileNet0.35-v2", "tf_graph" : "MobileNet0.35-v2-tf-graph",
            "output_names" : "Logits/Softmax", "input_names" : "input_1"},

        "MobileNet0.5-v2" :  {"path" : "./Models/MobileNet0.5-v2", "tf_graph" : "MobileNet0.5-v2-tf-graph",
            "output_names" : "Logits/Softmax", "input_names" : "input_1"},

        "MobileNet0.75-v2" :  {"path" : "./Models/MobileNet0.75-v2", "tf_graph" : "MobileNet0.75-v2-tf-graph",
            "output_names" : "Logits/Softmax", "input_names" : "input_1"},

        "MobileNet1.0-v2" :  {"path" : "./Models/MobileNet1.0-v2", "tf_graph" : "MobileNet1.0-v2-tf-graph",
            "output_names" : "Logits/Softmax", "input_names" : "input_1"},

        "MobileNet1.3-v2" :  {"path" : "./Models/MobileNet1.3-v2", "tf_graph" : "MobileNet1.3-v2-tf-graph",
            "output_names" : "Logits/Softmax", "input_names" : "input_1"},

        "MobileNet1.4-v2" :  {"path" : "./Models/MobileNet1.4-v2", "tf_graph" : "MobileNet1.4-v2-tf-graph",
            "output_names" : "Logits/Softmax", "input_names" : "input_1"},

        "MobileNet-small0.75-v3" :  {"path" : "./Models/MobileNet-small0.75-v3", "tf_graph" : "MobileNet-small0.75-v3-tf-graph",
            "output_names" : "reshape_11/Reshape", "input_names" : "input_1"},

        "MobileNet-small1.0-v3" :  {"path" : "./Models/MobileNet-small1.0-v3", "tf_graph" : "MobileNet-small1.0-v3-tf-graph",
            "output_names" : "reshape_11/Reshape", "input_names" : "input_1"},

        "MobileNet-small1.5-v3" :  {"path" : "./Models/MobileNet-small1.5-v3", "tf_graph" : "MobileNet-small1.5-v3-tf-graph",
            "output_names" : "reshape_11/Reshape", "input_names" : "input_1"},

        "MobileNet-large1.0-v3" :  {"path" : "./Models/MobileNet-large1.0-v3", "tf_graph" : "MobileNet-large1.0-v3-tf-graph",
            "output_names" : "reshape_10/Reshape", "input_names" : "input_1"},

        "NASNet-Mobile" : {"path" : "./Models/NASNet-Mobile", "tf_graph" : "NASNet-Mobile-tf-graph",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"}, 

        "NASNet-Large" : {"path" : "./Models/NASNet-Large", "tf_graph" : "NASNet-Large-tf-graph",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},

        "ResNet-20" : {"path" : "./Models/ResNet-20", "tf_graph" : "ResNet-20-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-32" : {"path" : "./Models/ResNet-32", "tf_graph" : "ResNet-32-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-44" : {"path" : "./Models/ResNet-44", "tf_graph" : "ResNet-44-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-56" : {"path" : "./Models/ResNet-56", "tf_graph" : "ResNet-56-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-110" : {"path" : "./Models/ResNet-110", "tf_graph" : "ResNet-110-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-164" : {"path" : "./Models/ResNet-164", "tf_graph" : "ResNet-164-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-200" : {"path" : "./Models/ResNet-200", "tf_graph" : "ResNet-200-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-20V2" : {"path" : "./Models/ResNet-20V2", "tf_graph" : "ResNet-20V2-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-38V2" : {"path" : "./Models/ResNet-38V2", "tf_graph" : "ResNet-38V2-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-47V2" : {"path" : "./Models/ResNet-47V2", "tf_graph" : "ResNet-47V2-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-56V2" : {"path" : "./Models/ResNet-56V2", "tf_graph" : "ResNet-56V2-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-110V2" : {"path" : "./Models/ResNet-110V2", "tf_graph" : "ResNet-110V2-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-164V2" : {"path" : "./Models/ResNet-164V2", "tf_graph" : "ResNet-164V2-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-1001V2" : {"path" : "./Models/ResNet-1001V2", "tf_graph" : "ResNet-1001V2-tf-graph",
            "output_names" : "dense_1/Softmax", "input_names" : "input_1"},

        "ResNet-18" : {"path" : "./Models/ResNet-18", "tf_graph" : "ResNet-18-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ResNet-34" : {"path" : "./Models/ResNet-34", "tf_graph" : "ResNet-34-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ResNet-50" : {"path" : "./Models/ResNet-50", "tf_graph" : "ResNet-50-tf-graph",
            "output_names" : "fc1000/Softmax", "input_names" : "input_1"},

        "ResNet-101" : {"path" : "./Models/ResNet-101", "tf_graph" : "ResNet-101-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "ResNet-152" : {"path" : "./Models/ResNet-152", "tf_graph" : "ResNet-152-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "ResNet-50V2" : {"path" : "./Models/ResNet-50V2", "tf_graph" : "ResNet-50V2-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "ResNet-101V2" : {"path" : "./Models/ResNet-101V2", "tf_graph" : "ResNet-101V2-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "ResNet-152V2" : {"path" : "./Models/ResNet-152V2", "tf_graph" : "ResNet-152V2-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "ResNext-50" : {"path" : "./Models/ResNext-50", "tf_graph" : "ResNext-50-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ResNext-101" : {"path" : "./Models/ResNext-101", "tf_graph" : "ResNext-101-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "SEResNet-18" : {"path" : "./Models/SEResNet-18", "tf_graph" : "SEResNet-18-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "SEResNet-34" : {"path" : "./Models/SEResNet-34", "tf_graph" : "SEResNet-34-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "SEResNet-50" : {"path" : "./Models/SEResNet-50", "tf_graph" : "SEResNet-50-tf-graph",
            "output_names" : "output/Softmax", "input_names" : "input_1"},

        "SEResNet-101" : {"path" : "./Models/SEResNet-101", "tf_graph" : "SEResNet-101-tf-graph",
            "output_names" : "output/Softmax", "input_names" : "input_1"},

        "SEResNet-152" : {"path" : "./Models/SEResNet-152", "tf_graph" : "SEResNet-152-tf-graph",
            "output_names" : "output/Softmax", "input_names" : "input_1"},

        "SEResNext-50" : {"path" : "./Models/SEResNext-50", "tf_graph" : "SEResNext-50-tf-graph",
            "output_names" : "output/Softmax", "input_names" : "input_1"},

        "SEResNext-101" : {"path" : "./Models/SEResNext-101", "tf_graph" : "SEResNext-101-tf-graph",
            "output_names" : "output/Softmax", "input_names" : "input_1"},

        "SENet-154" : {"path" : "./Models/SENet-154", "tf_graph" : "SENet-154-tf-graph",
            "output_names" : "output/Softmax", "input_names" : "input_1"},

        "SqueezeNet-v1.1" : {"path" : "./Models/SqueezeNet-v1.1", "tf_graph" : "SqueezeNet-v1.1-tf-graph",
            "output_names" : "loss/Softmax", "input_names" : "input_1"},

        "VGG-16" : {"path" : "./Models/VGG-16", "tf_graph" : "VGG-16-tf-graph",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},

        "VGG-19" : {"path" : "./Models/VGG-19", "tf_graph" : "VGG-19-tf-graph",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},

        "MNASNet0.35" : {"path" : "./Models/MNASNet0.35", "tf_graph" : "MNASNet0.35-tf-graph",
            "output_names" : "dense/Softmax", "input_names" : "input_1"},

        "MNASNet0.5" : {"path" : "./Models/MNASNet0.5", "tf_graph" : "MNASNet0.5-tf-graph",
            "output_names" : "dense/Softmax", "input_names" : "input_1"},

        "MNASNet0.75" : {"path" : "./Models/MNASNet0.75", "tf_graph" : "MNASNet0.75-tf-graph",
            "output_names" : "dense/Softmax", "input_names" : "input_1"},

        "MNASNet1.0" : {"path" : "./Models/MNASNet1.0", "tf_graph" : "MNASNet1.0-tf-graph",
            "output_names" : "dense/Softmax", "input_names" : "input_1"},

        "MNASNet1.4" : {"path" : "./Models/MNASNet1.4", "tf_graph" : "MNASNet1.4-tf-graph",
            "output_names" : "dense/Softmax", "input_names" : "input_1"},

        "EfficientNet-B0" : {"path" : "./Models/EfficientNet-B0", "tf_graph" : "EfficientNet-B0-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "EfficientNet-B1" : {"path" : "./Models/EfficientNet-B1", "tf_graph" : "EfficientNet-B1-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "EfficientNet-B2" : {"path" : "./Models/EfficientNet-B2", "tf_graph" : "EfficientNet-B2-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "EfficientNet-B3" : {"path" : "./Models/EfficientNet-B3", "tf_graph" : "EfficientNet-B3-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "EfficientNet-B4" : {"path" : "./Models/EfficientNet-B4", "tf_graph" : "EfficientNet-B4-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "EfficientNet-B5" : {"path" : "./Models/EfficientNet-B5", "tf_graph" : "EfficientNet-B5-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "EfficientNet-B6" : {"path" : "./Models/EfficientNet-B6", "tf_graph" : "EfficientNet-B6-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            
        "EfficientNet-B7" : {"path" : "./Models/EfficientNet-B7", "tf_graph" : "EfficientNet-B7-tf-graph",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},

        "ShuffleNet0.5-v1" :  {"path" : "./Models/ShuffleNet0.5-v1", "tf_graph" : "ShuffleNet0.5-v1-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ShuffleNet1.0-v1" :  {"path" : "./Models/ShuffleNet1.0-v1", "tf_graph" : "ShuffleNet1.0-v1-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ShuffleNet1.5-v1" :  {"path" : "./Models/ShuffleNet1.5-v1", "tf_graph" : "ShuffleNet1.5-v1-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ShuffleNet2.0-v1" :  {"path" : "./Models/ShuffleNet2.0-v1", "tf_graph" : "ShuffleNet2.0-v1-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ShuffleNet0.5-v2" :  {"path" : "./Models/ShuffleNet0.5-v2", "tf_graph" : "ShuffleNet0.5-v2-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ShuffleNet1.0-v2" :  {"path" : "./Models/ShuffleNet1.0-v2", "tf_graph" : "ShuffleNet1.0-v2-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ShuffleNet1.5-v2" :  {"path" : "./Models/ShuffleNet1.5-v2", "tf_graph" : "ShuffleNet1.5-v2-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        "ShuffleNet2.0-v2" :  {"path" : "./Models/ShuffleNet2.0-v2", "tf_graph" : "ShuffleNet2.0-v2-tf-graph",
            "output_names" : "softmax/Softmax", "input_names" : "input_1"},

        }
        
        #get the information from the passed args, 
        model_info = models_infos[args.model_name]
        model_name = args.model_name
        batch_size = int(args.batch_size)
        image_size = (int(args.image_size), int(args.image_size), 3)
        tf_graph_path = model_info['path']+'/Saved-Model/'+model_info['tf_graph']+'.pb'

        if (model_name == 'ResNet-20' or model_name == 'ResNet-32' or model_name == 'ResNet-44' or model_name == 'ResNet-56' or model_name == 'ResNet-110' or model_name == 'ResNet-164' or model_name == 'ResNet-200') :
            tf_graph_path = model_info['path']+'/Saved-Model/'+model_info['tf_graph']+'_'+args.image_size+'.pb'
        elif (model_name == 'ResNet-20V2' or model_name == 'ResNet-38V2' or model_name == 'ResNet-47V2' or model_name == 'ResNet-56V2' or model_name == 'ResNet-110V2' or model_name == 'ResNet-164V2' or model_name == 'ResNet-1001V2') :
            tf_graph_path = model_info['path']+'/Saved-Model/'+model_info['tf_graph']+'_'+args.image_size+'.pb'
        elif (model_name == 'ShuffleNet0.5-v1' or model_name == 'ShuffleNet1.0-v1' or model_name == 'ShuffleNet1.5-v1' or model_name == 'ShuffleNet2.0-v1') :
            tf_graph_path = model_info['path']+'/Saved-Model/'+model_info['tf_graph']+'_'+args.image_size+'.pb'
        elif (model_name == 'ShuffleNet0.5-v2' or model_name == 'ShuffleNet1.0-v2' or model_name == 'ShuffleNet1.5-v2' or model_name == 'ShuffleNet2.0-v2') :
            tf_graph_path = model_info['path']+'/Saved-Model/'+model_info['tf_graph']+'_'+args.image_size+'.pb'
        else :
            tf_graph_path = model_info['path']+'/Saved-Model/'+model_info['tf_graph']+'.pb'

    
        #deserialize the frozen graph
        TF_graph = tf.Graph()
        
        print('*********** Graph deserialization ************')
        tff_graph = get_frozen_graph(tf_graph_path)

        print('*********** Graph importation into TF graph ************')   
        tf.import_graph_def(tff_graph, name='')

        # input and output tensor names, output tensor name is a must for the prediction, to get the final results from the tf.graph
        # by specifying the output tensor name (ie the node in the graph)
        input_names = model_info['input_names']
        output_names = model_info['output_names']
        input_tensor_name = input_names + ":0"
        output_tensor_name = output_names + ":0"
        output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)  

        times_ = []

        with TF_graph.as_default():
            print("********* Start of the real measurments (110 predictions) **********")
            for i in range(0,110) :
                # Prapare image input for test
                index = str(i+1)
                img_path = './dataset/'+index+'.jpeg'
                x = image_to_tensor(img_path, image_size, args.model_name, False)
                img = []
                img.append(x)
                batch = np.array(img)
                feed_dict = {
                    input_tensor_name: batch
                }
                start = time.time()
                batch_prediction = tf_sess.run(output_tensor, feed_dict)
                end = time.time()
                delta = end - start
                print(start, end, delta)
                times_.append(delta)

            create_csv(args.model_name, args.batch_size, times_[10:], "./csv_files_"+args.target_device+"/"+args.image_size+"x"+args.image_size+"/"+args.model_name+"_"+str(batch_size)+"_"+args.image_size+".csv")
            with open('./execution_time.csv', 'a', newline='') as file :
                writer = csv.writer(file)
                writer.writerow([args.model_name, (int(args.image_size), int(args.image_size), 3), min(times_[10:]), max(times_[10:]), mean(times_[10:])])

            #mean_delta = np.array(times_).mean()
            #fps = 1 / mean_delta
            #print('inference time for the batch (ms): ', mean_delta*1000)
            #print('FPs = ', fps)

            print("********* End of the real measurments (110 predictions) **********")


#************************************ End of the main script ******************************************* #


