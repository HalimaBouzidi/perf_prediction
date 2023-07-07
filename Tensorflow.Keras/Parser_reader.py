import argparse
import json
import csv

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="MobileNet-v1", type=str, help="Specifiy the path of the csv file")
    parser.add_argument("image_size", default="75", type=str, help="Specifiy the image size")
    args = parser.parse_args()
    model_name = args.model_name
    input_shape = (1, int(args.image_size), int(args.image_size), 3)

    csv_file = "./Models/"+model_name+"/Saved-Model/"+model_name+"_"+args.image_size+".csv"

    layers_count = 0
    conv_count = 0
    sum_params = 0
    sum_activation_in = 0
    sum_activation_out = 0

    #Read the csv file
    with open(csv_file, newline='') as csvfile :
        csv_reader = csv.reader(csvfile)
        for idx, row in enumerate(csv_reader, start=0) :
            if (idx !=0) :
                layers_count += 1
                if(row[1]=='Conv2D' or row[1]=='SeparableConv2D' or row[1]=='DepthwiseConv2D' or row[1]=='Conv2DTranspose') :
                    conv_count += 1
                sum_params += int(row[11])
                sum_activation_in += int(row[3])
                sum_activation_out += int(row[10])
    
    print('number of keras layers : ', layers_count)
    print('number of Conv layers : ', conv_count)
    print('Sum of in activations :', sum_activation_in)
    print('Sum of out activations :', sum_activation_out)
    print('Number of parameters : ', sum_params)

    with open('./Model_parser_info.csv', 'a', newline='') as file :
        writer = csv.writer(file)
        writer.writerow([model_name, input_shape, layers_count, conv_count, sum_activation_in, sum_activation_out, sum_params])
