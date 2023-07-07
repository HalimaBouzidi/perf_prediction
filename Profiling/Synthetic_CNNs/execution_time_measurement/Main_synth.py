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
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf_config.gpu_options.allow_growth = True

# Create session and load graph
tf_sess = tf.Session(config=tf_config)
K.set_session(tf_sess)

	# ******************************* functions for the preprocessing and postprocessing ************************************* #

# the following function will transform a given image to an array of the specified size (image_size with the data layout (N, H, W, C))
def image_to_tensor(image_path, image_size, model_name, not_batch) :
    img = image.load_img(image_path, target_size=image_size[:2])
    x = image.img_to_array(img)
    if (not_batch) : 
        x = np.expand_dims(x, axis=0)
    
    from keras.applications.imagenet_utils import preprocess_input
    x = preprocess_input(x)

    return x

# The following function deserialize the frozen graph in order to import it to the tensorflow graph of the instantiate TF session.
def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk. (from the rep Saved-Model)"""
    with TF_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
    return graph_def

# The following function will create a csv file, which contains the execution times of the inferences.
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
        parser.add_argument("batch_size", default="1", type=str, help="Specifiy the batch size")
        parser.add_argument("target_device", default="Nano", type=str, help="Specifiy the target device")
        args = parser.parse_args()

        model_name = args.model_name
        batch_size = int(args.batch_size)
        split = model_name.split("_")
        input_s = split[2]
        image_size = (int(input_s), int(input_s), 3)
        tf_graph_path = './Generated_Models/Saved-Model/'+model_name+'-tf-graph.pb'

	#This is fixed for all of the synthetic CNNs
        model_input = 'input_1'
        model_output = 'softmax/Softmax'

        #deserialize the frozen graph of the inference
        TF_graph = tf.Graph()
        
        print('*********** Graph deserialization ************')
        tff_graph = get_frozen_graph(tf_graph_path)

        print('*********** Graph importation into TF graph ************')   
        tf.import_graph_def(tff_graph, name='')

        # input and output tensor names, output tensor name is necessary for the prediction, to get the final calculated result from the tf.graph
        # by specifying the output tensor name (ie the node in the graph)
        input_names = model_input
        output_names = model_output
        input_tensor_name = input_names + ":0"
        output_tensor_name = output_names + ":0"
        output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)  

        images = []

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

                images.append(feed_dict)

        with TF_graph.as_default():
            print("********* Start of the real measurments (110 predictions) **********")
            for i in range(0,110) :
                batch_prediction = tf_sess.run(output_tensor, images[i])
            print('fin')

            print("********* End of the real measurments (110 predictions) **********")


#************************************ End of the main script ******************************************* #


