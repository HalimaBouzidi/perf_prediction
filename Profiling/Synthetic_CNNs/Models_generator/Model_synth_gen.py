import argparse
import csv
import keras
import tensorflow as tf
import random
from keras import layers
from keras.models import Model
from keras.layers import Input
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("nb_models", default="MobileNet-v1", type=str, help="Specify the number of models")
    
    args = parser.parse_args()

    save_pb_dir = './Generated_Models/Saved-Model'

    nb_models = int(args.nb_models)

    input_sizes = [32, 56, 64, 75, 90, 100, 112, 128, 150, 200, 224, 240, 256, 300, 320, 331, 350, 400, 
    448, 480, 500, 512, 568, 600, 620, 700, 720, 750, 800, 820, 850, 900]

    filter_size =[1, 2, 3, 5, 7, 9, 11]

    nb_filters = [8, 16, 24, 32, 40, 48, 56, 64, 68, 72, 80, 96, 100, 112, 120, 128, 144, 150, 160, 192, 200, 224, 256,
    288, 300, 320, 331, 350, 384, 400, 450, 480, 500, 512, 620, 700, 720]

    strides = [1, 2, 3, 4, 5]

    bn_mp_avgp = [0, 1, 2]

    #This line must be executed before loading Keras model.
    tf.keras.backend.set_learning_phase(0) 
    
    for i in range(0, nb_models) :
        conv_tab = []
        after_conv = []
        top_tab = []
        input_s = random.choice(input_sizes)
        nb_conv = random.randint(4, 50)
        img_input = Input(shape=(input_s, input_s, 3))
        pool_position = random.randint(2, 5)
        print(img_input, nb_conv)
        for j in range(0, nb_conv) :
            if(j == 0) :
                conv_tab.append([])
                conv_tab[j].append(random.choice(nb_filters))
                conv_tab[j].append(random.choice(filter_size))
                x = layers.Conv2D(conv_tab[j][0], (conv_tab[j][1], conv_tab[j][1]),
                        activation='relu',
                        padding='same',
                        name='b_conv'+str(j+1))(img_input)
            else : 
                conv_tab.append([])
                conv_tab[j].append(random.choice(nb_filters))
                conv_tab[j].append(random.choice(filter_size))
                x = layers.Conv2D(conv_tab[j][0], (conv_tab[j][1], conv_tab[j][1]),
                        activation='relu',
                        padding='same',
                        name='b_conv'+str(j+1))(x)

            after_conv.append([])
            after_conv[j].append(random.randint(0, 1))
            ch_p = after_conv[j][0]
            if (ch_p == 0) :
                after_conv[j].append(-1)
                after_conv[j].append(-1)
                x = layers.BatchNormalization(name='bn_conv'+str(j+1))(x)

            elif (ch_p == 1) :
                if(j % pool_position == 0) :
                    after_conv[j].append(random.randint(2,3))
                    after_conv[j].append(random.choice(strides))
                    print('b_max_pool'+str(j+1), after_conv[j])
                    x = layers.MaxPooling2D((after_conv[j][1], after_conv[j][1]), 
                    strides=(after_conv[j][2], after_conv[j][2]), name='b_max_pool'+str(j+1))(x)
                else : 
                    after_conv[j].append(-1)
                    after_conv[j].append(-1)


                
        
        top_tab.append(random.randint(0, 2))
        ch_f = top_tab[0]
        if (ch_f == 0):
            x = layers.GlobalAveragePooling2D(name="global_pool")(x)
        elif(ch_f == 1):
            x = layers.GlobalMaxPooling2D(name="global_pool")(x)
        elif(ch_f == 2) :
            x = layers.Flatten(name='flatten')(x)

        top_tab.append(random.choice([10, 100, 200, 500, 1000]))
        x = layers.Dense(units=top_tab[1], name="fc")(x)
        x = layers.Activation('softmax', name='softmax')(x)
        print('top : ',top_tab)
        model_name = 'basic_model_'+str(input_s)+'_'+str(nb_conv)
        model = Model(img_input, x, name=model_name)
        model_frozen_graph  = model_name+'-tf-graph'
        saved_model_json = './Generated_Models/json_files/'+model_name+'.json'

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

        frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

        with open('./models_info_new_12.csv', 'a', newline='') as file :
            writer = csv.writer(file)
            writer.writerow([model_name, input_names, output_names])
        print(input_names, output_names)

        with open('./generated_models_12.csv', 'a', newline='') as file :
            writer = csv.writer(file)
            writer.writerow([input_s, nb_conv])
            for k in range(0, nb_conv) :
                writer.writerow(["conv_tab", conv_tab[k][0], conv_tab[k][1]])
                writer.writerow(["after_conv", after_conv[k][0], after_conv[k][1], after_conv[k][2]])
            writer.writerow(["top_tab", top_tab[0], top_tab[1]])
            writer.writerow(['', '', ''])
