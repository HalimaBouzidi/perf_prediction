import argparse
import tensorflow as tf
import keras.backend as K
import keras
import tensorflow.keras as tfkeras
from classification_models.keras import Classifiers
from keras.layers import Input
from tensorflow.keras.models import load_model

# Import Models
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras_contrib.applications.densenet import DenseNetImageNet161 as DenseNet161
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet101, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras_squeezenet import SqueezeNet
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="MobileNet-v1", type=str, help="Specifiy the name of the model")
    parser.add_argument("image_size", default="MobileNet-v1", type=str, help="Specifiy the name of the model")
    args = parser.parse_args()
    
    model_name = args.model_name
    image_size = int(args.image_size)

    if(model_name == 'DenseNet-121' or model_name == 'DenseNet-161' or model_name == 'DenseNet-169' or model_name == 'DenseNet-201' or model_name == 'DenseNet-264') :
        input_tensor = Input(shape=(image_size, image_size, 3))
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
        input_tensor = Input(shape=(image_size, image_size, 3))
        if (model_name == 'Inception-v1') :
            from Inception_v1 import InceptionV1
            model = InceptionV1(input_shape=(image_size, image_size, 3), weights=None)
        elif (model_name == 'Inception-v3') :
            model = keras.applications.inception_v3.InceptionV3(input_tensor=input_tensor, weights=None)
        elif (model_name == 'InceptionResNetV2') :
            model = keras.applications.inception_resnet_v2.InceptionResNetV2(input_tensor=input_tensor, weights=None)
        elif (model_name == 'Xception') :
            model = keras.applications.xception.Xception(input_tensor=input_tensor, weights=None)

    elif (model_name == 'MobileNet0.25-v1' or model_name == 'MobileNet0.5-v1'  or model_name == 'MobileNet0.75-v1' or model_name == 'MobileNet1.0-v1' ) :
        input_tensor = Input(shape=(image_size, image_size, 3))
        if (model_name == 'MobileNet0.25-v1') :
            model = keras.applications.mobilenet.MobileNet(alpha=0.25, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet0.5-v1') :
            model = keras.applications.mobilenet.MobileNet(alpha=0.5, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet0.75-v1') :
            model = keras.applications.mobilenet.MobileNet(alpha=0.75, input_tensor=input_tensor, weights=None)
        elif (model_name == 'MobileNet1.0-v1') :
            model = keras.applications.mobilenet.MobileNet(alpha=1.0, input_tensor=input_tensor, weights=None)

    elif (model_name == 'MobileNet0.35-v2' or model_name == 'MobileNet0.5-v2'  or model_name == 'MobileNet0.75-v2' or model_name == 'MobileNet1.0-v2' or model_name == 'MobileNet1.3-v2'or model_name == 'MobileNet1.4-v2' ) :
        input_tensor = Input(shape=(image_size, image_size, 3))
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
            f = MobileNetV3_Small(shape=(image_size,image_size,3), n_class=1000, alpha=0.75, include_top=True)
            model = f.build()

        if(model_name == 'MobileNet-small1.0-v3') :
            from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
            f = MobileNetV3_Small(shape=(image_size,image_size,3), n_class=1000, alpha=1.0, include_top=True)
            model = f.build()

        if(model_name == 'MobileNet-small1.5-v3') :
            from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
            f = MobileNetV3_Small(shape=(image_size,image_size,3), n_class=1000, alpha=1.5, include_top=True)
            model = f.build()

        elif(model_name == 'MobileNet-large1.0-v3') :
            from mobilenet_v3.mobilenet_v3_large import MobileNetV3_Large
            f = MobileNetV3_Large(shape=(image_size,image_size,3), n_class=1000, alpha=1.0, include_top=True)
            model = f.build()


    elif (model_name == 'ResNet-18') :
        input_tensor = Input(shape=(image_size, image_size, 3))
        ResNet18 = Classifiers.get('resnet18')[0]
        model = ResNet18(input_tensor=input_tensor, weights=None)

    elif (model_name == 'ResNet-34') :
        input_tensor = Input(shape=(image_size, image_size, 3))
        ResNet34 = Classifiers.get('resnet34')[0]
        model = ResNet34(input_tensor=input_tensor, weights=None)

    elif (model_name == 'ResNet-50' or model_name == 'ResNet-101' or model_name == 'ResNet-152') :
        input_tensor = Input(shape=(image_size, image_size, 3))
        if (model_name == 'ResNet-50') :
            from keras.applications.resnet50 import ResNet50
            model = ResNet50(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNet-101') :
            model = keras.applications.resnet.ResNet101(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNet-152') :
            model = keras.applications.resnet.ResNet152(input_tensor=input_tensor, weights=None)

    elif (model_name == 'ResNet-50V2' or model_name == 'ResNet-101V2' or model_name == 'ResNet-152V2') :
        input_tensor = Input(shape=(image_size, image_size, 3))
        if (model_name == 'ResNet-50V2') :
            model = keras.applications.resnet_v2.ResNet50V2(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNet-101V2') :
            model = keras.applications.resnet_v2.ResNet101V2(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNet-152V2') :
            model = keras.applications.resnet_v2.ResNet152V2(input_tensor=input_tensor, weights=None)

    elif (model_name == 'SEResNet-18' or model_name == 'SEResNet-34' or model_name == 'SEResNet-50' or model_name == 'SEResNet-101' or model_name == 'SEResNet-152') :
        input_tensor = Input(shape=(image_size, image_size, 3))
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
        input_tensor = Input(shape=(image_size, image_size, 3))
        if (model_name == 'ResNext-50') :
            ResNext50, preprocess_input = Classifiers.get('resnext50')
            model = ResNext50(input_tensor=input_tensor, weights=None)
        elif (model_name == 'ResNext-101') :
            ResNext101 = Classifiers.get('resnext101')[0]
            model = ResNext101(input_tensor=input_tensor, weights=None)

    elif (model_name == 'SEResNext-50' or model_name == 'SEResNext-101') :   
        input_tensor = Input(shape=(image_size, image_size, 3))
        if (model_name == 'SEResNext-50') :
            SEResNeXt50 = Classifiers.get('seresnext50')[0]
            model = SEResNeXt50(input_tensor=input_tensor, weights=None)
        elif (model_name == 'SEResNext-101') :
            SEResNeXt101 = Classifiers.get('seresnext101')[0]
            model = SEResNeXt101(input_tensor=input_tensor, weights=None)

    elif (model_name == 'NASNet-Mobile' or model_name == 'NASNet-Large') :
        input_tensor = Input(shape=(240, 240, 3))
        if (model_name == 'NASNet-Mobile') :
            NASNetMobile = Classifiers.get('nasnetmobile')[0]
            model = NASNetMobile(input_tensor=input_tensor, weights=None)
        elif (model_name == 'NASNet-Large') :
            NASNetMobile = Classifiers.get('nasnetlarge')[0]
            model = NASNetMobile(input_tensor=input_tensor, weights=None)

    elif (model_name == 'VGG-16' or model_name == 'VGG-19') :
        input_tensor = Input(shape=(240, 240, 3))
        if (model_name == 'VGG-16') :
            from keras.applications.vgg16 import VGG16
            model = VGG16(input_tensor=input_tensor, weights=None)
        elif (model_name == 'VGG-19') :
            from keras.applications.vgg19 import VGG19
            model = VGG19(input_tensor=input_tensor, weights=None)

    elif (model_name == 'SENet-154') :
        input_tensor = Input(shape=(image_size, image_size, 3))
        SENet154 = Classifiers.get('senet154')[0]
        model = SENet154(input_tensor=input_tensor, weights=None)

    elif (model_name == 'SqueezeNet-v1.1') :
        input_tensor = Input(shape=(image_size, image_size, 3))
        from keras_squeezenet import SqueezeNet
        model = SqueezeNet(input_tensor=input_tensor, weights=None)

    elif (model_name == 'EfficientNet-B0' or model_name == 'EfficientNet-B1' or model_name == 'EfficientNet-B2' or model_name == 'EfficientNet-B3' or model_name == 'EfficientNet-B4' or model_name == 'EfficientNet-B5' or model_name == 'EfficientNet-B6' or model_name == 'EfficientNet-B7') :
        import tensorflow.keras as tfkeras
        input_tensor_efficient = tfkeras.layers.Input(shape=(image_size, image_size, 3))
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
        elif (model_name == 'EfficientNet-B5') :
            from efficientnet.tfkeras import EfficientNetB5
            model = EfficientNetB5(input_tensor=input_tensor_efficient, weights=None)
        elif (model_name == 'EfficientNet-B6') :
            from efficientnet.tfkeras import EfficientNetB6
            model = EfficientNetB6(input_tensor=input_tensor_efficient, weights=None)
        elif (model_name == 'EfficientNet-B7') :
            from efficientnet.tfkeras import EfficientNetB7
            model = EfficientNetB7(input_tensor=input_tensor_efficient, weights=None)

    elif (model_name == 'MNASNet0.35' or model_name == 'MNASNet0.5' or model_name == 'MNASNet0.75' or model_name == 'MNASNet1.0' or model_name == 'MNASNet1.4') :
        from MNASNet import MnasNet
        if (model_name == 'MNASNet0.35') :
            model = MnasNet(input_shape=(image_size, image_size, 3), alpha=0.35)
        elif (model_name == 'MNASNet0.5') :
            model = MnasNet(input_shape=(image_size, image_size, 3), alpha=0.5)
        elif (model_name == 'MNASNet0.75') :
            model = MnasNet(input_shape=(image_size, image_size, 3), alpha=0.75)
        elif (model_name == 'MNASNet1.0') :
            model = MnasNet(input_shape=(image_size, image_size, 3), alpha=1.0)
        elif (model_name == 'MNASNet1.4') :
            model = MnasNet(input_shape=(image_size, image_size, 3), alpha=1.4)

    elif (model_name == 'DPN-92' or model_name == 'DPN-98' or model_name == 'DPN-107' or model_name == 'DPN-137') :
        from dual_path_network import DPN92, DPN98, DPN107, DPN137
        input_tensor = Input(shape=(image_size, image_size, 3))
        if (model_name == 'DPN-92') :
            model = DPN92(input_tensor=input_tensor)
        elif (model_name == 'DPN-98') :
            model = DPN98(input_tensor=input_tensor)
        elif (model_name == 'DPN-107') :
            model = DPN107(input_tensor=input_tensor)
        elif (model_name == 'DPN-137') :
            model = DPN137(input_tensor=input_tensor)

    elif (model_name == 'GhostNet') :
        from ghostnet_ import GhostModel
        ghost = GhostModel(1000,image_size,3)
        model = ghost.model

    elif (model_name == 'ResNet-20') :
        from resnet_cifare_10 import resnet_v1  
        model= resnet_v1(input_shape=(image_size, image_size, 3), depth=20, num_classes=10)
    elif (model_name == 'ResNet-32') :
        from resnet_cifare_10 import resnet_v1
        model= resnet_v1(input_shape=(image_size, image_size, 3), depth=32, num_classes=10)
    elif (model_name == 'ResNet-44') :
        from resnet_cifare_10 import resnet_v1
        model= resnet_v1(input_shape=(image_size, image_size, 3), depth=44, num_classes=10)
    elif (model_name == 'ResNet-56') :
        from resnet_cifare_10 import resnet_v1
        model= resnet_v1(input_shape=(image_size, image_size, 3), depth=56, num_classes=10)
    elif (model_name == 'ResNet-110') :
        from resnet_cifare_10 import resnet_v1
        model= resnet_v1(input_shape=(image_size, image_size, 3), depth=110, num_classes=10)
    elif (model_name == 'ResNet-164') :
        from resnet_cifare_10 import resnet_v1
        model= resnet_v1(input_shape=(image_size, image_size, 3), depth=164, num_classes=10)
    elif (model_name == 'ResNet-200') :
        from resnet_cifare_10 import resnet_v1
        model= resnet_v1(input_shape=(image_size, image_size, 3), depth=200, num_classes=10)

    elif (model_name == 'ResNet-20V2') :
        from resnet_cifare_10 import resnet_v2
        model = resnet_v2(input_shape=(image_size, image_size, 3), depth=20, num_classes=10)
    elif (model_name == 'ResNet-38V2') :
        from resnet_cifare_10 import resnet_v2
        model = resnet_v2(input_shape=(image_size, image_size, 3), depth=38, num_classes=10)
    elif (model_name == 'ResNet-47V2') :
        from resnet_cifare_10 import resnet_v2
        model = resnet_v2(input_shape=(image_size, image_size, 3), depth=47, num_classes=10)
    elif (model_name == 'ResNet-56V2') :
        from resnet_cifare_10 import resnet_v2
        model = resnet_v2(input_shape=(image_size, image_size, 3), depth=56, num_classes=10)
    elif (model_name == 'ResNet-110V2') :
        from resnet_cifare_10 import resnet_v2
        model = resnet_v2(input_shape=(image_size, image_size, 3), depth=110, num_classes=10)
    elif (model_name == 'ResNet-164V2') :
        from resnet_cifare_10 import resnet_v2
        model = resnet_v2(input_shape=(image_size, image_size, 3), depth=164, num_classes=10)
    elif (model_name == 'ResNet-1001V2') :
        from resnet_cifare_10 import resnet_v2
        model = resnet_v2(input_shape=(
            image_size, image_size, 3), depth=1001, num_classes=10)


    model_json = model.to_json()
    with open("./Models/"+model_name+"/json_files/"+model_name+"_"+args.image_size+".json", "w") as json_file:
        json_file.write(model_json)

