#!/usr/bin/env python
# coding: utf-8

# In[20]:


import argparse
import json
import csv
import math
import efficientnet.tfkeras
import keras.backend
from keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from mobilenet_v3.mobilenet_v3_small import MobileNetV3_Small
from mobilenet_v3.mobilenet_v3_large import MobileNetV3_Large
from mobilenet_v3.mobilenet_base import MobileNetBase


#from keras.models import model_from_json


# In[33]:


def create_csv(layers, output_path) :
    activ_in_size= 1
    activ_out_size= 1
    with open(output_path, 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(["layer_name", 'class_name', "input_shape", "activation_in_size", "d_type","Filters"
                         ,"Filter_size","Activ_fun" ,"use_B", "output_shape", "activation_out_size", "Params"])
        for idx,layer in enumerate(layers, start=0) :
            layer = json.dumps(layer)
            layer = json.loads(layer)
            for idx_, s in enumerate(layer["input_shape"], start=0) :
                if (idx_ != 0 and idx !=1 ) :
                    activ_in_size *= s
            for idx_, s in enumerate(layer["output_shape"], start=0) :
                if idx_ != 0 :
                    activ_out_size *= s
            writer.writerow([layer["layer_name"], layer["class_name"], layer["input_shape"], activ_in_size,
            layer["d_type"], layer["Filters"], layer["Filter_size"], layer["Activ_fun"], layer["use_B"], 
            layer["output_shape"], activ_out_size, layer["Params"]])
            activ_in_size= 1
            activ_out_size= 1


# In[23]:


def csv_line(layer_class_name, layer_name, config) :
    data = {}
    
    data['layer_name'] = layer_name #layer name
    
    data['class_name'] = layer_class_name #layer class name
    
    data['input_shape'] = [] # input shape
    
    data['d_type'] = config['dtype'] # precision type (32 bits, 64 bits...etc)
    
    if (is_json_key_present(config, 'filters')):
        data['Filters'] = config['filters'] # Nb of filters
    else :
        data['Filters'] = 'NaN'
        
    if (is_json_key_present(config, 'kernel_size')):
        data['Filter_size'] = config['kernel_size'] # filter size
    else :
        data['Filter_size'] = 'NaN'
    
    if (is_json_key_present(config, 'activation')):
        data['Activ_fun'] = config['activation']  #activation function if used
    else :
        data['Activ_fun'] = 'NaN'
    
    if (is_json_key_present(config, 'use_bias')):
        data['use_B'] = config['use_bias'] # bias if used
    else :
        data['use_B'] = 'NaN'
    
    data['output_shape'] = []
    data['Params'] = 0 # Nb of parametrs (0 for initialization only, cuz we'll get the information from another way)
    
    return data


# In[24]:

def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def is_json_key_present(json, key):
    try:
        buf = json[key]
    except KeyError:
        return False

    return True


# In[25]:

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="MobileNet-v1", type=str, help="Specifiy the name of the model")
    parser.add_argument("image_size", default="MobileNet-v1", type=str, help="Specifiy the name of the model")
    args = parser.parse_args()

    model_name = args.model_name
    image_size_ = int(args.image_size)

    # My idea is to generate first a json file which parse the CNN architecture descriptor, 
    # then use that json file descriptor to extract all the informations about all the layers of the model, 
    # such as their name, class_name, configuration, parametres and shapes of input and output

    # load the model architecture from the saved json file and create model
    input_path = "./Models/"+model_name+"/json_files/"+model_name+"_"+args.image_size+".json"
    output_path = "./Models/"+model_name+"/csv_files/"+model_name+"_"+args.image_size+".csv"

    json_file = open(input_path, 'r')
    loaded_model_json = json_file.read()
    # print(loaded_model_json)
    json_file.close()

    model_parsed = json.loads(loaded_model_json)
    model = model_from_json(loaded_model_json)

    #model.summary() # in order to check after

    #3 instantiate stuff :p
    layers = []
    jsn_obj = {}
    first = True


    # In[34]:


    #4 create a json object of conv layers with the aviabale parametrs from the model descriptor
    vect = model_parsed['config']['layers']
    for layer in vect :
        #print(layer)
        #extract all the layers configurations from the model descriptor    
        jsn_obj = csv_line(layer['class_name'], layer['name'], layer['config'])
        layers.append(jsn_obj) 

    for idx,layer in enumerate(layers, start=0) : 
        layer = json.dumps(layer)
        layer = json.loads(layer)
        
        layer['Params'] = model.get_layer(layer['layer_name']).count_params()
        layer['output_shape'] = keras.backend.int_shape(model.get_layer(layer['layer_name']).output)
        
        if(idx == 0) :
            layer['input_shape'] = model.input_shape
        else :
            layer['input_shape'] = input_next
        input_next = keras.backend.int_shape(model.get_layer(layer['layer_name']).output)
        layers[idx] = layer
        
    create_csv(layers, output_path)
# Finally go check the csv file in the path mentionned above


# In[ ]:




