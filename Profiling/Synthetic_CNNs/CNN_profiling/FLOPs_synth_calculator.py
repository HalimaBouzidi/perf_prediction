import argparse
import tensorflow as tf
import keras.backend as K
from keras import layers
from keras.models import Model
import csv

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("idx", default="1", type=str, help="Specifiy the idx of the model")
    args = parser.parse_args()

    idx = int(args.idx)
    lines = []
    
    with open('./generated_models_12.csv', newline='') as csvfile :
        csv_reader = csv.reader(csvfile)
        csv_data = list(csv_reader)
        line = csv_data[idx]
        while (idx <= 1773 and line[0] != ''):
            lines.append(line)
            line = csv_data[idx]
            idx = idx+1

    print(lines)
    
    model_name = 'basic_model_'+str(lines[0][0])+'_'+str(lines[0][1])
    input_shape = (1, int(lines[0][0]), int(lines[0][0]), 3)
    print(input_shape)
    input_sh = (int(lines[0][0]), int(lines[0][0]), 3)

    lines.pop(0)
    lines.pop(0)

    run_meta = tf.RunMetadata()
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)

        img_input = layers.Input(tensor=tf.ones(shape=input_shape, dtype='float32'))

        for idx, row in enumerate(lines, start=0) :
            if(row[0] == 'conv_tab') : 
                if(idx == 0) :
                    x = layers.Conv2D(int(row[1]), (int(row[2]), int(row[2])),
                    activation='relu',
                    padding='same',
                    name='b_conv'+str(idx+1))(img_input)
                else : 
                    x = layers.Conv2D(int(row[1]), (int(row[2]), int(row[2])),
                    activation='relu',
                    padding='same',
                    name='b_conv'+str(idx+1))(x)
            elif (row[0] == 'after_conv') :
                if(int(row[1]) == 0) :
                    x = layers.BatchNormalization(name='bn_conv'+str(idx+1))(x)
                elif(int(row[1]) == 1) :
                    if(int(row[2])!=-1 and int(row[3])!=-1) :
                        x = layers.MaxPooling2D((int(row[2]), int(row[2])), 
                    strides=(int(row[3]), int(row[3])), name='b_max_pool'+str(idx+1))(x)
            elif (row[0] == 'top_tab') :
                if(int(row[1]) == 0) :
                    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
                elif(int(row[1]) == 1) :
                    x = layers.GlobalMaxPooling2D(name="global_pool")(x)
                elif(int(row[1]) == 2) :
                    x = layers.Flatten(name='flatten')(x)

                x = layers.Dense(units=int(row[2]), name="fc")(x)
                x = layers.Activation('softmax', name='softmax')(x)

                break

        net = Model(img_input, x, name=model_name)

        net.summary()

        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("FLOPs : {:,} --- Parameters : {:,}".format(flops.total_float_ops, params.total_parameters))

        with open('./Models_FLOPS.csv', 'a', newline='') as file :
            writer = csv.writer(file)
            writer.writerow([model_name, input_shape, flops.total_float_ops, params.total_parameters])

