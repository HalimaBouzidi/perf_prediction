import argparse
import tensorflow as tf
import keras.backend as K
import keras
import csv
from classification_models.keras import Classifiers

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="MobileNet-v1", type=str, help="Specifiy the name of the model")
    parser.add_argument("image_size", default="75", type=str, help="Specifiy the image size")
    args = parser.parse_args()
    
    model_name = args.model_name
    input_shape = (1, int(args.image_size), int(args.image_size), 3)

    run_meta = tf.RunMetadata()
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)

        if(model_name == 'DenseNet-121') :
            from keras.applications.densenet import DenseNet121
            net = DenseNet121(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'DenseNet-161') :
            from keras_contrib.applications.densenet import DenseNetImageNet161 as DenseNet161
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = DenseNet161(input_tensor=input_tensor, weights=None)

        elif (model_name == 'DenseNet-169') :
            from keras.applications.densenet import DenseNet169
            net = DenseNet169(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif (model_name == 'DenseNet-201') :
            from keras.applications.densenet import DenseNet201
            net = DenseNet201(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif(model_name == 'DenseNet-264'):
            from keras_contrib.applications.densenet import DenseNetImageNet264 as DenseNet264
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net= DenseNet264(input_tensor = input_tensor, weights=None)

        if (model_name == 'Inception-v1') :
            from Inception_v1 import InceptionV1
            net = InceptionV1(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)  

        elif (model_name == 'Inception-v3') :
            from keras.applications.inception_v3 import InceptionV3
            net = InceptionV3(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'InceptionResNetV2') :
            from keras.applications.inception_resnet_v2 import InceptionResNetV2
            net = InceptionResNetV2(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'Xception') :
            from keras.applications.xception import Xception
            net = Xception(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'MobileNet0.25-v1') :
            from keras.applications.mobilenet import MobileNet
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = MobileNet(alpha=0.25, input_tensor = input_tensor, weights=None) 

        elif (model_name == 'MobileNet0.5-v1') :
            from keras.applications.mobilenet import MobileNet
            net = MobileNet(alpha=0.5, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif (model_name == 'MobileNet0.75-v1') :
            from keras.applications.mobilenet import MobileNet
            net = MobileNet(alpha=0.75, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif (model_name == 'MobileNet1.0-v1') :
            from keras.applications.mobilenet import MobileNet
            net = MobileNet(alpha=1.0, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif (model_name == 'MobileNet0.35-v2') :
            from keras.applications.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(alpha=0.35, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'MobileNet0.5-v2') :
            from keras.applications.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(alpha=0.5, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif (model_name == 'MobileNet0.75-v2') :
            from keras.applications.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(alpha=0.75, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif (model_name == 'MobileNet1.0-v2') :
            from keras.applications.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(alpha=1.0, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif (model_name == 'MobileNet1.3-v2') :
            from keras.applications.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(alpha=1.3, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif (model_name == 'MobileNet1.4-v2') :
            from keras.applications.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(alpha=1.4, input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None)

        elif(model_name == 'MobileNet-small0.75-v3') :
            from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
            f = MobileNetV3_Small(shape=input_shape, n_class=1000, alpha=0.75, include_top=True)
            net = f.build()

        elif(model_name == 'MobileNet-small1.0-v3') :
            from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
            f = MobileNetV3_Small(shape=input_shape, n_class=1000, alpha=1.0, include_top=True)
            net = f.build()

        elif(model_name == 'MobileNet-small1.5-v3') :
            from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
            f = MobileNetV3_Small(shape=input_shape, n_class=1000, alpha=1.5, include_top=True)
            net = f.build()

        elif(model_name == 'MobileNet-large1.0-v3') :
            from mobilenet_v3.mobilenet_v3_large import MobileNetV3_Large
            f = MobileNetV3_Large(shape=input_shape, n_class=1000, alpha=1.0, include_top=True)
            net = f.build()

        elif (model_name == 'ResNet-18') :
            ResNet18 = Classifiers.get('resnet18')[0]
            net = ResNet18(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNet-34') :
            ResNet34 = Classifiers.get('resnet34')[0]
            net = ResNet34(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SEResNet-18') :
            SEResNet18 = Classifiers.get('seresnet18')[0]
            net = SEResNet18(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SEResNet-34') :
            SEResNet34 = Classifiers.get('seresnet34')[0]
            net = SEResNet34(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SEResNet-50') :
            SEResNet50 = Classifiers.get('seresnet50')[0]
            net = SEResNet50(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SEResNet-101') :
            SEResNet101 = Classifiers.get('seresnet101')[0]
            net = SEResNet101(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SEResNet-152') :
            SEResNet152 = Classifiers.get('seresnet152')[0]
            net = SEResNet152(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNet-50') :
            from keras.applications.resnet50 import ResNet50
            net = ResNet50(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNet-101') :
            from keras.applications.resnet import ResNet101
            net = ResNet101(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNet-152') :
            from keras.applications.resnet import ResNet152
            net = ResNet152(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNet-50V2') :
            from keras.applications.resnet_v2 import ResNet50V2
            net = ResNet50V2(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNet-101V2') :
            from keras.applications.resnet_v2 import ResNet101V2
            net = ResNet101V2(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNet-152V2') :
            from keras.applications.resnet_v2 import ResNet152V2
            net = ResNet152V2(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNext-50') :
            ResNext50 = Classifiers.get('resnext50')[0]
            net = ResNext50(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'ResNext-101') :
            ResNext101 = Classifiers.get('resnext101')[0]
            net = ResNext101(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SEResNext-50') :
            SEResNeXt50 = Classifiers.get('seresnext50')[0]
            net = SEResNeXt50(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SEResNext-101') :
            SEResNeXt101 = Classifiers.get('seresnext101')[0]
            net = SEResNeXt101(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SENet-154') :
            SENet154 = Classifiers.get('senet154')[0]
            net = SENet154(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'SqueezeNet-v1.1') :
            from keras_squeezenet import SqueezeNet
            net = SqueezeNet(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'EfficientNet-B0') :
            from efficientnet.tfkeras import EfficientNetB0
            net = EfficientNetB0(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'EfficientNet-B1') :
            from efficientnet.tfkeras import EfficientNetB1
            net = EfficientNetB1(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'EfficientNet-B2') :
            from efficientnet.tfkeras import EfficientNetB2
            net = EfficientNetB2(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'EfficientNet-B3') :
            from efficientnet.tfkeras import EfficientNetB3
            net = EfficientNetB3(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'EfficientNet-B4') :
            from efficientnet.tfkeras import EfficientNetB4
            net = EfficientNetB4(input_tensor = tf.ones(shape=input_shape, dtype='float32'), weights=None) 

        elif (model_name == 'MNASNet0.35') :
            from MNASNet import MnasNet
            net = MnasNet(full_shape=input_shape, alpha=0.35)

        elif (model_name == 'MNASNet0.5') :
            from MNASNet import MnasNet
            net = MnasNet(full_shape=input_shape, alpha=0.5)

        elif (model_name == 'MNASNet0.75') :
            from MNASNet import MnasNet
            net = MnasNet(full_shape=input_shape, alpha=0.75)

        elif (model_name == 'MNASNet1.0') :
            from MNASNet import MnasNet
            net = MnasNet(full_shape=input_shape, alpha=1.0)

        elif (model_name == 'MNASNet1.4') :
            from MNASNet import MnasNet
            net = MnasNet(full_shape=input_shape, alpha=1.4)

        elif (model_name == 'DPN-92') :
            from dual_path_network import DPN92
            net = DPN92(input_tensor=tf.ones(shape=input_shape, dtype='float32'))
        elif (model_name == 'DPN-98') :
            from dual_path_network import DPN98
            net = DPN98(input_tensor=tf.ones(shape=input_shape, dtype='float32'))

        elif (model_name == 'DPN-107') :
            from dual_path_network import DPN107
            net = DPN107(input_tensor=tf.ones(shape=input_shape, dtype='float32'))

        elif (model_name == 'DPN-137') :
            from dual_path_network import DPN137
            net = DPN137(input_tensor=tf.ones(shape=input_shape, dtype='float32'))

        elif (model_name == 'ShuffleNet0.5-v1') :
            from shufflenet_v1 import ShuffleNet
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = ShuffleNet(include_top=True, input_tensor=input_tensor, scale_factor=0.5, bottleneck_ratio=1)

        elif (model_name == 'ShuffleNet1.0-v1') :
            from shufflenet_v1 import ShuffleNet
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = ShuffleNet(include_top=True, input_tensor=input_tensor, scale_factor=1.0, bottleneck_ratio=1)

        elif (model_name == 'ShuffleNet1.5-v1') :
            from shufflenet_v1 import ShuffleNet
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = ShuffleNet(include_top=True, input_tensor=input_tensor, scale_factor=1.5, bottleneck_ratio=1)

        elif (model_name == 'ShuffleNet2.0-v1') :
            from shufflenet_v1 import ShuffleNet
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = ShuffleNet(include_top=True, input_tensor=input_tensor, scale_factor=2.0, bottleneck_ratio=1)

        elif (model_name == 'ShuffleNet0.5-v2') :
            from shufflenetv2 import ShuffleNetV2
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = ShuffleNetV2(include_top=True, input_tensor=input_tensor, scale_factor=0.5, bottleneck_ratio=1)

        elif (model_name == 'ShuffleNet1.0-v2') :
            from shufflenetv2 import ShuffleNetV2
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = ShuffleNetV2(include_top=True, input_tensor=input_tensor, scale_factor=1.0, bottleneck_ratio=1)

        elif (model_name == 'ShuffleNet1.5-v2') :
            from shufflenetv2 import ShuffleNetV2
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = ShuffleNetV2(include_top=True, input_tensor=input_tensor, scale_factor=1.5, bottleneck_ratio=1)

        elif (model_name == 'ShuffleNet2.0-v2') :
            from shufflenetv2 import ShuffleNetV2
            from keras.layers import Input
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = ShuffleNetV2(include_top=True, input_tensor=input_tensor, scale_factor=2.0, bottleneck_ratio=1)

        elif (model_name == 'ResNet-20') :
            from resnet_cifare_10 import resnet_v1  
            net = resnet_v1(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=20, num_classes=10)
        elif (model_name == 'ResNet-32') :
            from resnet_cifare_10 import resnet_v1
            net = resnet_v1(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=32, num_classes=10)
        elif (model_name == 'ResNet-44') :
            from resnet_cifare_10 import resnet_v1
            net = resnet_v1(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=44, num_classes=10)
        elif (model_name == 'ResNet-56') :
            from resnet_cifare_10 import resnet_v1
            net = resnet_v1(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=56, num_classes=10)
        elif (model_name == 'ResNet-110') :
            from resnet_cifare_10 import resnet_v1
            net = resnet_v1(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=110, num_classes=10)
        elif (model_name == 'ResNet-164') :
            from resnet_cifare_10 import resnet_v1
            net = resnet_v1(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=164, num_classes=10)
        elif (model_name == 'ResNet-200') :
            from resnet_cifare_10 import resnet_v1
            net = resnet_v1(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=200, num_classes=10)

        elif (model_name == 'ResNet-20V2') :
            from resnet_cifare_10 import resnet_v2
            model = resnet_v2(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=20, num_classes=10)
        elif (model_name == 'ResNet-38V2') :
            from resnet_cifare_10 import resnet_v2
            model = resnet_v2(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=38, num_classes=10)
        elif (model_name == 'ResNet-47V2') :
            from resnet_cifare_10 import resnet_v2
            model = resnet_v2(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=47, num_classes=10)
        elif (model_name == 'ResNet-56V2') :
            from resnet_cifare_10 import resnet_v2
            model = resnet_v2(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=56, num_classes=10)
        elif (model_name == 'ResNet-110V2') :
            from resnet_cifare_10 import resnet_v2
            model = resnet_v2(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=110, num_classes=10)
        elif (model_name == 'ResNet-164V2') :
            from resnet_cifare_10 import resnet_v2
            model = resnet_v2(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape, depth=164, num_classes=10)

        elif (model_name == 'VGG-16') :
            from vgg16_ import VGG16
            net = VGG16(input_tensor = None, input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape) 

        elif (model_name == 'VGG-19') :
            from vgg19_ import VGG19
            net = VGG19(input_tensor = None, input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape) 

        elif (model_name == 'Basic_3') :
            from basic import BASIC_3
            model = BASIC_3(input_tensor = None, input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape)

        elif (model_name == 'Basic_4') :
            from basic import BASIC_4
            model = BASIC_4(input_tensor = None, input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape)

        elif (model_name == 'AlexNet') :
            from AlexNet import AlexNet
            model = AlexNet(input_shape=(int(args.image_size), int(args.image_size), 3), full_shape=input_shape)

        elif (model_name == 'NASNet-Mobile') :
            from keras.applications.nasnet import NASNetMobile
            from keras.layers import Input
            NASNetMobile = Classifiers.get('nasnetmobile')[0]
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = NASNetMobile(input_tensor = input_tensor) 

        elif (model_name == 'NASNet-Large') :
            from keras.applications.nasnet import NASNetLarge
            from keras.layers import Input
            NASNetLarge = Classifiers.get('nasnetlarge')[0]
            input_tensor = Input(tensor=tf.ones(shape=input_shape, dtype='float32'))
            net = NASNetLarge(input_tensor = input_tensor) 

        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("FLOPs : {:,} --- Parameters : {:,}".format(flops.total_float_ops, params.total_parameters))

        with open('./Models_FLOPS.csv', 'a', newline='') as file :
            writer = csv.writer(file)
            writer.writerow([model_name, input_shape, flops.total_float_ops, params.total_parameters])

