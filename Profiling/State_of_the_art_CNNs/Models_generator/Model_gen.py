import argparse
import csv
import keras
import tensorflow as tf
from keras.layers import Input
import tensorflow.keras as tfkeras
from classification_models.keras import Classifiers
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="MobileNet-v1", type=str, help="Specifiy the name of the model")
    parser.add_argument("input_size", default="28", type=str, help="Specifiy the input size for ShuffleNet")
    args = parser.parse_args()

    models_infos = { 
        "DenseNet-121" : {"path": "./Models/DenseNet-121","tf_graph" : "DenseNet-121-tf-graph"},
        "DenseNet-161" : {"path": "./Models/DenseNet-161","tf_graph" : "DenseNet-161-tf-graph"},
        "DenseNet-169" : {"path": "./Models/DenseNet-169","tf_graph" : "DenseNet-169-tf-graph"},
        "DenseNet-201" : {"path": "./Models/DenseNet-201","tf_graph" : "DenseNet-201-tf-graph"},
        "DenseNet-264" : {"path": "./Models/DenseNet-264","tf_graph" : "DenseNet-264-tf-graph"},
        "DPN-92" : {"path": "./Models/DPN-92","tf_graph" : "DPN-92-tf-graph"},
        "DPN-98" : {"path": "./Models/DPN-98","tf_graph" : "DPN-98-tf-graph"},
        "DPN-107" : {"path": "./Models/DPN-107","tf_graph" : "DPN-107-tf-graph"},
        "DPN-137" : {"path": "./Models/DPN-137","tf_graph" : "DPN-137-tf-graph"},
        "Inception-v1" : {"path" : "./Models/Inception-v1", "tf_graph" : "Inception-v1-tf-graph"},
        "Inception-v3" : {"path" : "./Models/Inception-v3", "tf_graph" : "Inception-v3-tf-graph"},
        "Xception" : {"path" : "./Models/Xception", "tf_graph" : "Xception-tf-graph"},
        "InceptionResNetV2" : {"path" : "./Models/InceptionResNetV2", "tf_graph" : "InceptionResNetV2-tf-graph"},
        "MobileNet0.25-v1" : {"path" : "./Models/MobileNet0.25-v1", "tf_graph" : "MobileNet0.25-v1-tf-graph"},
        "MobileNet0.5-v1" : {"path" : "./Models/MobileNet0.5-v1", "tf_graph" : "MobileNet0.5-v1-tf-graph"},
        "MobileNet0.75-v1" : {"path" : "./Models/MobileNet0.75-v1", "tf_graph" : "MobileNet0.75-v1-tf-graph"},
        "MobileNet1.0-v1" : {"path" : "./Models/MobileNet1.0-v1", "tf_graph" : "MobileNet1.0-v1-tf-graph"},
        "MobileNet0.35-v2" :  {"path" : "./Models/MobileNet0.35-v2", "tf_graph" : "MobileNet0.35-v2-tf-graph"},
        "MobileNet0.5-v2" :  {"path" : "./Models/MobileNet0.5-v2", "tf_graph" : "MobileNet0.5-v2-tf-graph"},
        "MobileNet0.75-v2" :  {"path" : "./Models/MobileNet0.75-v2", "tf_graph" : "MobileNet0.75-v2-tf-graph"},
        "MobileNet1.0-v2" :  {"path" : "./Models/MobileNet1.0-v2", "tf_graph" : "MobileNet1.0-v2-tf-graph"},
        "MobileNet1.3-v2" :  {"path" : "./Models/MobileNet1.3-v2", "tf_graph" : "MobileNet1.3-v2-tf-graph"},
        "MobileNet1.4-v2" :  {"path" : "./Models/MobileNet1.4-v2", "tf_graph" : "MobileNet1.4-v2-tf-graph"},
        "MobileNet-small0.75-v3" :  {"path" : "./Models/MobileNet-small0.75-v3", "tf_graph" : "MobileNet-small0.75-v3-tf-graph"},
        "MobileNet-small1.0-v3" :  {"path" : "./Models/MobileNet-small1.0-v3", "tf_graph" : "MobileNet-small1.0-v3-tf-graph"},
        "MobileNet-small1.5-v3" :  {"path" : "./Models/MobileNet-small1.5-v3", "tf_graph" : "MobileNet-small1.5-v3-tf-graph"},
        "MobileNet-large1.0-v3" :  {"path" : "./Models/MobileNet-large1.0-v3", "tf_graph" : "MobileNet-large1.0-v3-tf-graph"},
        "ShuffleNet0.5-v1" :  {"path" : "./Models/ShuffleNet0.5-v1", "tf_graph" : "ShuffleNet0.5-v1-tf-graph"},
        "ShuffleNet1.0-v1" :  {"path" : "./Models/ShuffleNet1.0-v1", "tf_graph" : "ShuffleNet1.0-v1-tf-graph"},
        "ShuffleNet1.5-v1" :  {"path" : "./Models/ShuffleNet1.5-v1", "tf_graph" : "ShuffleNet1.5-v1-tf-graph"},
        "ShuffleNet2.0-v1" :  {"path" : "./Models/ShuffleNet2.0-v1", "tf_graph" : "ShuffleNet2.0-v1-tf-graph"},
        "ShuffleNet0.5-v2" :  {"path" : "./Models/ShuffleNet0.5-v2", "tf_graph" : "ShuffleNet0.5-v2-tf-graph"},
        "ShuffleNet1.0-v2" :  {"path" : "./Models/ShuffleNet1.0-v2", "tf_graph" : "ShuffleNet1.0-v2-tf-graph"},
        "ShuffleNet1.5-v2" :  {"path" : "./Models/ShuffleNet1.5-v2", "tf_graph" : "ShuffleNet1.5-v2-tf-graph"},
        "ShuffleNet2.0-v2" :  {"path" : "./Models/ShuffleNet2.0-v2", "tf_graph" : "ShuffleNet2.0-v2-tf-graph"},
        "MNASNet0.35" : {"path" : "./Models/MNASNet0.35", "tf_graph" : "MNASNet0.35-tf-graph"}, 
        "MNASNet0.5" : {"path" : "./Models/MNASNet0.5", "tf_graph" : "MNASNet0.5-tf-graph"}, 
        "MNASNet0.75" : {"path" : "./Models/MNASNet0.75", "tf_graph" : "MNASNet0.75-tf-graph"}, 
        "MNASNet1.0" : {"path" : "./Models/MNASNet1.0", "tf_graph" : "MNASNet1.0-tf-graph"},  
        "MNASNet1.4" : {"path" : "./Models/MNASNet1.4", "tf_graph" : "MNASNet1.4-tf-graph"},        
        "ResNet-18" : {"path" : "./Models/ResNet-18", "tf_graph" : "ResNet-18-tf-graph"},
        "ResNet-20" : {"path" : "./Models/ResNet-20", "tf_graph" : "ResNet-20-tf-graph"},
        "ResNet-32" : {"path" : "./Models/ResNet-32", "tf_graph" : "ResNet-32-tf-graph"},
        "ResNet-34" : {"path" : "./Models/ResNet-34", "tf_graph" : "ResNet-34-tf-graph"},
        "ResNet-44" : {"path" : "./Models/ResNet-44", "tf_graph" : "ResNet-44-tf-graph"},
        "ResNet-50" : {"path" : "./Models/ResNet-50", "tf_graph" : "ResNet-50-tf-graph"},
        "ResNet-56" : {"path" : "./Models/ResNet-56", "tf_graph" : "ResNet-56-tf-graph"},
        "ResNet-101" : {"path" : "./Models/ResNet-101", "tf_graph" : "ResNet-101-tf-graph"},
        "ResNet-110" : {"path" : "./Models/ResNet-110", "tf_graph" : "ResNet-110-tf-graph"},
        "ResNet-152" : {"path" : "./Models/ResNet-152", "tf_graph" : "ResNet-152-tf-graph"},
        "ResNet-164" : {"path" : "./Models/ResNet-164", "tf_graph" : "ResNet-164-tf-graph"},
        "ResNet-200" : {"path" : "./Models/ResNet-200", "tf_graph" : "ResNet-200-tf-graph"},
        "ResNet-20V2" : {"path" : "./Models/ResNet-20V2", "tf_graph" : "ResNet-20V2-tf-graph"},
        "ResNet-38V2" : {"path" : "./Models/ResNet-38V2", "tf_graph" : "ResNet-38V2-tf-graph"},
        "ResNet-47V2" : {"path" : "./Models/ResNet-47V2", "tf_graph" : "ResNet-47V2-tf-graph"},
        "ResNet-50V2" : {"path" : "./Models/ResNet-50V2", "tf_graph" : "ResNet-50V2-tf-graph"},
        "ResNet-56V2" : {"path" : "./Models/ResNet-56V2", "tf_graph" : "ResNet-56V2-tf-graph"},
        "ResNet-101V2" : {"path" : "./Models/ResNet-101V2", "tf_graph" : "ResNet-101V2-tf-graph"},
        "ResNet-110V2" : {"path" : "./Models/ResNet-110V2", "tf_graph" : "ResNet-110V2-tf-graph"},
        "ResNet-152V2" : {"path" : "./Models/ResNet-152V2", "tf_graph" : "ResNet-152V2-tf-graph"},
        "ResNet-164V2" : {"path" : "./Models/ResNet-164V2", "tf_graph" : "ResNet-164V2-tf-graph"},
        "ResNext-50" : {"path" : "./Models/ResNext-50", "tf_graph" : "ResNext-50-tf-graph"},
        "ResNext-101" : {"path" : "./Models/ResNext-101", "tf_graph" : "ResNext-101-tf-graph"},
        "SEResNet-18" : {"path" : "./Models/SEResNet-18", "tf_graph" : "SEResNet-18-tf-graph"},
        "SEResNet-34" : {"path" : "./Models/SEResNet-34", "tf_graph" : "SEResNet-34-tf-graph"},
        "SEResNet-50" : {"path" : "./Models/SEResNet-50", "tf_graph" : "SEResNet-50-tf-graph"},
        "SEResNet-101" : {"path" : "./Models/SEResNet-101", "tf_graph" : "SEResNet-101-tf-graph"},
        "SEResNet-152" : {"path" : "./Models/SEResNet-152", "tf_graph" : "SEResNet-152-tf-graph"},
        "SEResNext-50" : {"path" : "./Models/SEResNext-50", "tf_graph" : "SEResNext-50-tf-graph"},
        "SEResNext-101" : {"path" : "./Models/SEResNext-101", "tf_graph" : "SEResNext-101-tf-graph"},
        "SENet-154" : {"path" : "./Models/SENet-154", "tf_graph" : "SENet-154-tf-graph"},
        "SqueezeNet-v1.1" : {"path" : "./Models/SqueezeNet-v1.1", "tf_graph" : "SqueezeNet-v1.1-tf-graph"},
        "EfficientNet-B0" : {"path" : "./Models/EfficientNet-B0", "tf_graph" : "EfficientNet-B0-tf-graph"},
        "EfficientNet-B1" : {"path" : "./Models/EfficientNet-B1", "tf_graph" : "EfficientNet-B1-tf-graph"},
        "EfficientNet-B2" : {"path" : "./Models/EfficientNet-B2", "tf_graph" : "EfficientNet-B2-tf-graph"},
        "EfficientNet-B3" : {"path" : "./Models/EfficientNet-B3", "tf_graph" : "EfficientNet-B3-tf-graph"},
        "EfficientNet-B4" : {"path" : "./Models/EfficientNet-B4", "tf_graph" : "EfficientNet-B4-tf-graph"},
        "NASNet-Mobile" : {"path" : "./Models/NASNet-Mobile", "tf_graph" : "NASNet-Mobile-tf-graph"},                                                                                                                                                      
        "NASNet-Large" : {"path" : "./Models/NASNet-Large", "tf_graph" : "NASNet-Large-tf-graph"},
        "VGG-16" : {"path" : "./Models/VGG-16", "tf_graph" : "VGG-16-tf-graph"},
        "VGG-19" : {"path" : "./Models/VGG-19", "tf_graph" : "VGG-19-tf-graph"},
        "Basic_3" : {"path" : "./Models/Basic_3", "tf_graph" : "Basic_3-tf-graph"},
        "Basic_4" : {"path" : "./Models/Basic_4", "tf_graph" : "Basic_4-tf-graph"},
        "AlexNet" : {"path" : "./Models/AlexNet", "tf_graph" : "AlexNet-tf-graph"}

        }
    
    #get the information from the passed args, 
    model_info = models_infos[args.model_name]
    model_name = args.model_name
    saved_model_path = model_info['path']+'/Saved-Model/'+args.model_name+'.h5'

    save_pb_dir = model_info['path']+'/Saved-Model'

    model_frozen_graph = model_info['tf_graph']

    save_pb_dir = model_info['path']+'/Saved-Model'

    # This line must be executed before loading Keras model.
    tf.keras.backend.set_learning_phase(0) 


    if(model_name == 'DenseNet-121' or model_name == 'DenseNet-161' or model_name == 'DenseNet-169' or model_name == 'DenseNet-201' or model_name == 'DenseNet-264') :
        input_tensor = Input(shape=(None, None, 3))
        if(model_name == 'DenseNet-121'):
            model = keras.applications.densenet.DenseNet121(input_tensor=input_tensor, weights=None)
        elif(model_name == 'DenseNet-161'):
            from keras_contrib.applications.densenet import DenseNetImageNet161
            model = DenseNetImageNet161(input_tensor=input_tensor, weights=None)
        elif(model_name == 'DenseNet-169'):
            model = keras.applications.densenet.DenseNet169(input_tensor=input_tensor, weights=None)
        elif(model_name == 'DenseNet-201'):
            model = keras.applications.densenet.DenseNet201(input_tensor=input_tensor, weights=None)  
        elif(model_name == 'DenseNet-264'):
            from keras_contrib.applications.densenet import DenseNetImageNet264
            model = DenseNetImageNet264(input_tensor=input_tensor, weights=None) 

    elif (model_name == 'Inception-v1' or model_name == 'Inception-v3' or model_name == 'InceptionResNetV2' or model_name == 'Xception') :
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'Inception-v1') :
            from Inception_v1 import InceptionV1
            model = InceptionV1(input_tensor=input_tensor, weights=None)
        elif (model_name == 'Inception-v3') :
            model = keras.applications.inception_v3.InceptionV3(input_tensor=input_tensor, weights=None)
        elif (model_name == 'InceptionResNetV2') :
            model = keras.applications.inception_resnet_v2.InceptionResNetV2(input_tensor=input_tensor, weights=None)
        elif (model_name == 'Xception') :
            model = keras.applications.xception.Xception(input_tensor=input_tensor, weights=None)

    elif (model_name == 'MobileNet0.25-v1' or model_name == 'MobileNet0.5-v1'  or model_name == 'MobileNet0.75-v1' or model_name == 'MobileNet1.0-v1' ) :
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'MobileNet0.25-v1') :
            model = keras.applications.mobilenet.MobileNet(alpha=0.25, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet0.5-v1') :
            model = keras.applications.mobilenet.MobileNet(alpha=0.5, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet0.75-v1') :
            model = keras.applications.mobilenet.MobileNet(alpha=0.75, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet1.0-v1') :
            model = keras.applications.mobilenet.MobileNet(alpha=1.0, input_tensor=input_tensor, weights=None)

    elif (model_name == 'MobileNet0.35-v2' or model_name == 'MobileNet0.5-v2'  or model_name == 'MobileNet0.75-v2' or model_name == 'MobileNet1.0-v2' or model_name == 'MobileNet1.3-v2'or model_name == 'MobileNet1.4-v2' ) :
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'MobileNet0.35-v2') :
            model = keras.applications.mobilenet_v2.MobileNetV2(alpha=0.35, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet0.5-v2') :
            model = keras.applications.mobilenet_v2.MobileNetV2(alpha=0.5, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet0.75-v2') :
            model = keras.applications.mobilenet_v2.MobileNetV2(alpha=0.75, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet1.0-v2') :
            model = keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet1.3-v2') :
            model = keras.applications.mobilenet_v2.MobileNetV2(alpha=1.3, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet1.4-v2') :
            model = keras.applications.mobilenet_v2.MobileNetV2(alpha=1.4, input_tensor=input_tensor, weights=None)
        
    elif (model_name == 'MobileNet-small0.75-v3' or model_name == 'MobileNet-small1.0-v3' or model_name == 'MobileNet-small1.5-v3' or model_name == 'MobileNet-large1.0-v3') :
        if(model_name == 'MobileNet-small0.75-v3') :
            from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
            f = MobileNetV3_Small(shape=(None,None,3), n_class=1000, alpha=0.75, include_top=True)
            model = f.build()

        if(model_name == 'MobileNet-small1.0-v3') :
            from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
            f = MobileNetV3_Small(shape=(None,None,3), n_class=1000, alpha=1.0, include_top=True)
            model = f.build()

        if(model_name == 'MobileNet-small1.5-v3') :
            from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
            f = MobileNetV3_Small(shape=(None,None,3), n_class=1000, alpha=1.5, include_top=True)
            model = f.build()

        elif(model_name == 'MobileNet-large1.0-v3') :
            from mobilenet_v3.mobilenet_v3_large import MobileNetV3_Large
            f = MobileNetV3_Large(shape=(None,None,3), n_class=1000, alpha=1.0, include_top=True)
            model = f.build()


    elif (model_name == 'ResNet-18') :
        input_tensor = Input(shape=(None, None, 3))
        ResNet18 = Classifiers.get('resnet18')[0]
        model = ResNet18(input_tensor=input_tensor, weights=None)

    elif (model_name == 'ResNet-34') :
        input_tensor = Input(shape=(None, None, 3))
        ResNet34 = Classifiers.get('resnet34')[0]
        model = ResNet34(input_tensor=input_tensor, weights=None)

    elif (model_name == 'ResNet-50' or model_name == 'ResNet-101' or model_name == 'ResNet-152') :
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'ResNet-50') :
            from keras.applications.resnet50 import ResNet50
            model = ResNet50(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNet-101') :
            model = keras.applications.resnet.ResNet101(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNet-152') :
            model = keras.applications.resnet.ResNet152(input_tensor=input_tensor, weights=None)

    elif (model_name == 'ResNet-50V2' or model_name == 'ResNet-101V2' or model_name == 'ResNet-152V2') :
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'ResNet-50V2') :
            model = keras.applications.resnet_v2.ResNet50V2(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNet-101V2') :
            model = keras.applications.resnet_v2.ResNet101V2(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNet-152V2') :
            model = keras.applications.resnet_v2.ResNet152V2(input_tensor=input_tensor, weights=None)

    elif (model_name == 'SEResNet-18' or model_name == 'SEResNet-34' or model_name == 'SEResNet-50' or model_name == 'SEResNet-101' or model_name == 'SEResNet-152') :
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'SEResNet-18') :
            SEResNet18 = Classifiers.get('seresnet18')[0]
            model = SEResNet18(input_tensor=input_tensor, weights=None)
        elif (model_name == 'SEResNet-34') :
            SEResNet34 = Classifiers.get('seresnet34')[0]
            model = SEResNet34(input_tensor=input_tensor, weights=None)
        elif (model_name == 'SEResNet-50') :
            SEResNet50 = Classifiers.get('seresnet50')[0]
            model = SEResNet50(input_tensor=input_tensor, weights=None)
        elif (model_name == 'SEResNet-101') :
            SEResNet101 = Classifiers.get('seresnet101')[0]
            model = SEResNet101(input_tensor=input_tensor, weights=None)
        elif (model_name == 'SEResNet-152') :
            SEResNet152 = Classifiers.get('seresnet152')[0]
            model = SEResNet152(input_tensor=input_tensor, weights=None)
    
    elif (model_name == 'ResNext-50' or model_name == 'ResNext-101') :
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'ResNext-50') :
            ResNext50, preprocess_input = Classifiers.get('resnext50')
            model = ResNext50(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNext-101') :
            ResNext101 = Classifiers.get('resnext101')[0]
            model = ResNext101(input_tensor=input_tensor, weights=None)

    elif (model_name == 'SEResNext-50' or model_name == 'SEResNext-101') :   
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'SEResNext-50') :
            SEResNeXt50 = Classifiers.get('seresnext50')[0]
            model = SEResNeXt50(input_tensor=input_tensor, weights=None)
        elif (model_name == 'SEResNext-101') :
            SEResNeXt101 = Classifiers.get('seresnext101')[0]
            model = SEResNeXt101(input_tensor=input_tensor, weights=None)

    elif (model_name == 'NASNet-Mobile' or model_name == 'NASNet-Large') :
        input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
        if (model_name == 'NASNet-Mobile') :
            NASNetMobile = Classifiers.get('nasnetmobile')[0]
            model = NASNetMobile(input_tensor=input_tensor, weights=None)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'NASNet-Large') :
            NASNetMobile = Classifiers.get('nasnetlarge')[0]
            model = NASNetMobile(input_tensor=input_tensor, weights=None)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'

    elif (model_name == 'VGG-16' or model_name == 'VGG-19') :
        input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
        if (model_name == 'VGG-16') :
            from keras.applications.vgg16 import VGG16
            model = VGG16(input_tensor=input_tensor, weights=None)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'VGG-19') :
            from keras.applications.vgg19 import VGG19
            model = VGG19(input_tensor=input_tensor, weights=None)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'

    elif (model_name == 'Basic_3') :
        input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
        from basic import BASIC_3
        model = BASIC_3(input_tensor=input_tensor, weights=None)
        model_frozen_graph  = model_frozen_graph+'_'+args.input_size
        saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'

    elif (model_name == 'Basic_4') :
        input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
        from basic import BASIC_4
        model = BASIC_4(input_tensor=input_tensor, weights=None)
        model_frozen_graph  = model_frozen_graph+'_'+args.input_size
        saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'

    elif (model_name == 'AlexNet') :
        from AlexNet import AlexNet
        model = AlexNet(input_shape=(int(args.input_size), int(args.input_size), 3))
        model_frozen_graph  = model_frozen_graph+'_'+args.input_size
        saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'

    elif (model_name == 'SENet-154') :
        input_tensor = Input(shape=(None, None, 3))
        SENet154 = Classifiers.get('senet154')[0]
        model = SENet154(input_tensor=input_tensor, weights=None)

    elif (model_name == 'SqueezeNet-v1.1') :
        input_tensor = Input(shape=(None, None, 3))
        from keras_squeezenet import SqueezeNet
        model = SqueezeNet(input_tensor=input_tensor, weights=None)

    elif (model_name == 'EfficientNet-B0' or model_name == 'EfficientNet-B1' or model_name == 'EfficientNet-B2' or model_name == 'EfficientNet-B3' or model_name == 'EfficientNet-B4') :
        import tensorflow.keras as tfkeras
        input_tensor_efficient = tfkeras.layers.Input(shape=(None, None, 3))
        if (model_name == 'EfficientNet-B0') :
            from efficientnet.tfkeras import EfficientNetB0
            model = EfficientNetB0(input_tensor=input_tensor_efficient, weights=None)
        elif (model_name == 'EfficientNet-B1') :
            from efficientnet.tfkeras import EfficientNetB1
            model = EfficientNetB1(input_tensor=input_tensor_efficient, weights=None)
        elif (model_name == 'EfficientNet-B2') :
            from efficientnet.tfkeras import EfficientNetB2
            model = EfficientNetB2(input_tensor=input_tensor_efficient, weights=None)
        elif (model_name == 'EfficientNet-B3') :
            from efficientnet.tfkeras import EfficientNetB3
            model = EfficientNetB3(input_tensor=input_tensor_efficient, weights=None)
        elif (model_name == 'EfficientNet-B4') :
            from efficientnet.tfkeras import EfficientNetB4
            model = EfficientNetB4(input_tensor=input_tensor_efficient, weights=None)

    elif (model_name == 'MNASNet0.35' or model_name == 'MNASNet0.5' or model_name == 'MNASNet0.75' or model_name == 'MNASNet1.0' or model_name == 'MNASNet1.4') :
        from MNASNet import MnasNet
        if (model_name == 'MNASNet0.35') :
            model = MnasNet(input_shape=(None, None, 3), alpha=0.35)
        elif (model_name == 'MNASNet0.5') :
            model = MnasNet(input_shape=(None, None, 3), alpha=0.5)
        elif (model_name == 'MNASNet0.75') :
            model = MnasNet(input_shape=(None, None, 3), alpha=0.75)
        elif (model_name == 'MNASNet1.0') :
            model = MnasNet(input_shape=(None, None, 3), alpha=1.0)
        elif (model_name == 'MNASNet1.4') :
            model = MnasNet(input_shape=(None, None, 3), alpha=1.4)

    elif (model_name == 'DPN-92' or model_name == 'DPN-98' or model_name == 'DPN-107' or model_name == 'DPN-137') :
        from dual_path_network import DPN92, DPN98, DPN107, DPN137
        input_tensor = Input(shape=(None, None, 3))
        if (model_name == 'DPN-92') :
            model = DPN92(input_tensor=input_tensor)
        elif (model_name == 'DPN-98') :
            model = DPN98(input_tensor=input_tensor)
        elif (model_name == 'DPN-107') :
            model = DPN107(input_tensor=input_tensor)
        elif (model_name == 'DPN-137') :
            model = DPN137(input_tensor=input_tensor)
    
    elif (model_name == 'ShuffleNet0.5-v2' or model_name == 'ShuffleNet1.0-v2' or model_name == 'ShuffleNet1.5-v2' or model_name == 'ShuffleNet2.0-v2') :
        from shufflenetv2 import ShuffleNetV2
        if (model_name == 'ShuffleNet0.5-v2') :
            #input_tensor = Input(input_shape=(int(args.input_size), int(args.input_size), 3))
            model = ShuffleNetV2(include_top=True, input_shape=(int(args.input_size), int(args.input_size), 3), scale_factor=0.5, bottleneck_ratio=1)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ShuffleNet1.0-v2') :
            #input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
            model = ShuffleNetV2(include_top=True, input_shape=(int(args.input_size), int(args.input_size), 3), scale_factor=1.0, bottleneck_ratio=1)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ShuffleNet1.5-v2') :
            #input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
            model = ShuffleNetV2(include_top=True, input_shape=(int(args.input_size), int(args.input_size), 3), scale_factor=1.5, bottleneck_ratio=1)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ShuffleNet2.0-v2') :
            #input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
            model = ShuffleNetV2(include_top=True, input_shape=(int(args.input_size), int(args.input_size), 3), scale_factor=2.0, bottleneck_ratio=1)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'

    elif (model_name == 'ShuffleNet0.5-v1' or model_name == 'ShuffleNet1.0-v1' or model_name == 'ShuffleNet1.5-v1' or model_name == 'ShuffleNet2.0-v1') :
        from shufflenet_v1 import ShuffleNet
        if (model_name == 'ShuffleNet0.5-v1') :
            input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
            model = ShuffleNet(include_top=True, input_tensor=input_tensor, scale_factor=0.5, bottleneck_ratio=1)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ShuffleNet1.0-v1') :
            input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
            model = ShuffleNet(include_top=True, input_tensor=input_tensor, scale_factor=1.0, bottleneck_ratio=1)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ShuffleNet1.5-v1') :
            input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
            model = ShuffleNet(include_top=True, input_tensor=input_tensor, scale_factor=1.5, bottleneck_ratio=1)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ShuffleNet2.0-v1') :
            input_tensor = Input(shape=(int(args.input_size), int(args.input_size), 3))
            model = ShuffleNet(include_top=True, input_tensor=input_tensor, scale_factor=2.0, bottleneck_ratio=1)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'

    elif (model_name == 'ResNet-20' or model_name == 'ResNet-32' or model_name == 'ResNet-44' or model_name == 'ResNet-56' or model_name == 'ResNet-110' or model_name == 'ResNet-164' or model_name == 'ResNet-164' or model_name == 'ResNet-200') :
        from resnet_cifare_10 import resnet_v1
        if (model_name == 'ResNet-20') :
            model = resnet_v1(input_shape=(int(args.input_size), int(args.input_size), 3), depth=20, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-32') :
            model = resnet_v1(input_shape=(int(args.input_size), int(args.input_size), 3), depth=32, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-44') :
            model = resnet_v1(input_shape=(int(args.input_size), int(args.input_size), 3), depth=44, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-56') :
            model = resnet_v1(input_shape=(int(args.input_size), int(args.input_size), 3), depth=56, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-110') :
            model = resnet_v1(input_shape=(int(args.input_size), int(args.input_size), 3), depth=110, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-164') :
            model = resnet_v1(input_shape=(int(args.input_size), int(args.input_size), 3), depth=164, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-200') :
            model = resnet_v1(input_shape=(int(args.input_size), int(args.input_size), 3), depth=200, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
    
    elif (model_name == 'ResNet-20V2' or model_name == 'ResNet-38V2' or model_name == 'ResNet-47V2' or model_name == 'ResNet-56V2' or model_name == 'ResNet-110V2' or model_name == 'ResNet-164V2') :
        from resnet_cifare_10 import resnet_v2
        if (model_name == 'ResNet-20V2') :
            model = resnet_v2(input_shape=(int(args.input_size), int(args.input_size), 3), depth=20, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-38V2') :
            model = resnet_v2(input_shape=(int(args.input_size), int(args.input_size), 3), depth=38, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-47V2') :
            model = resnet_v2(input_shape=(int(args.input_size), int(args.input_size), 3), depth=47, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-56V2') :
            model = resnet_v2(input_shape=(int(args.input_size), int(args.input_size), 3), depth=56, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-110V2') :
            model = resnet_v2(input_shape=(int(args.input_size), int(args.input_size), 3), depth=110, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'
        elif (model_name == 'ResNet-164V2') :
            model = resnet_v2(input_shape=(int(args.input_size), int(args.input_size), 3), depth=164, num_classes=10)
            model_frozen_graph  = model_frozen_graph+'_'+args.input_size
            saved_model_json = model_info['path']+'/json_files/'+args.model_name+'_'+args.input_size+'.json'

    model_json = model.to_json()
    with open(saved_model_json, "w") as json_file:
        json_file.write(model_json)

    def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name=model_frozen_graph+'.pb', save_pb_as_text=False):
        with graph.as_default():
            graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
            graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
            return graphdef_frozen


    session = tf.keras.backend.get_session()

    input_names = [t.op.name for t in model.inputs]
    output_names = [t.op.name for t in model.outputs]

    with open('./models_info.csv', 'a', newline='') as file :
        writer = csv.writer(file)
        writer.writerow([model_name, input_names, output_names])
    print(input_names, output_names)

    frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)
