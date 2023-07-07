import argparse
import json
import csv

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="MobileNet-v1", type=str, help="Specifiy the path of the csv file")
    args = parser.parse_args()
    model_name = args.model_name
    split = model_name.split("_")
    input_s = split[2]
    input_shape = (1, int(input_s), int(input_s), 3)

    csv_file = "./Generated_Models/csv_files/"+model_name+".csv"

    layers_count = 0
    conv_count = 0
    fc_count = 0
    bn_count = 0
    pooling_count =0
    flatten = 0
    global_avg_pooling = 0
    sum_params_conv = 0
    sum_params_fc = 0
    sum_params_bn = 0
    sum_activation = 0
    total_params = 0
    weighted_sum_neurones = 0
    fc_multiply = 0 
    conv_multiply = 0

    #Read the csv file
    with open(csv_file, newline='') as csvfile :
        csv_reader = csv.reader(csvfile)
        for idx, row in enumerate(csv_reader, start=0) :
            if (idx !=0) :
                layers_count += 1
                if(row[1]=='Conv2D' or row[1]=='Conv2DTranspose') :
                    conv_count += 1
                    split_1 = row[2].split(" ")
                    depth = int(split_1[3].split("]")[0])
                    split_1 = row[6].split(" ")
                    kernel_size = int(split_1[1].split("]")[0])
                    weighted_sum_neurones += int(row[10])*kernel_size*kernel_size*depth
                    sum_params_conv += int(row[11])

                elif (row[1]=='SeparableConv2D') :
                    conv_count += 1
                    split_1 = row[2].split(" ")
                    depth_1 = int(split_1[3].split("]")[0])
                    width_1 = int(split_1[1].split(",")[0])
                    split_2 = row[9].split(" ")
                    depth_2 = int(split_2[3].split("]")[0])
                    width_2 = int(split_2[1].split(",")[0])
                    split_3 = row[6].split(" ")
                    kernel_size = int(split_3[1].split("]")[0])
                    weighted_sum_neurones += ((width_1*width_2*kernel_size*depth)+(width_2*width_2*kernel_size*depth))*int(row[5])
                    sum_params_conv += int(row[11])

                elif(row[1]=='Dense'):
                    fc_count +=1
                    sum_params_fc += int(row[11])
                    fc_multiply += (int(row[3])*int(row[10]))

                elif(row[1]=='BatchNormalization') : 
                    bn_count +=1
                    sum_params_bn += int(row[11]) # other params like batch normalization

                elif(row[1]=='MaxPooling2D' or row[1]=='AveragePooling2D') :
                    pooling_count +=1

                if(row[1]=='Flatten') :
                    flatten = 1
                elif(row[1]=='GlobalAveragePooling2D') :
                    global_avg_pooling = 1

                sum_activation += int(row[10])
                total_params += int(row[11])

    print(sum_params_conv, sum_params_bn, sum_params_fc, total_params)

    with open('./Model_parser_info_new_12.csv', 'a', newline='') as file :
        writer = csv.writer(file)
        writer.writerow([model_name, input_shape, layers_count, conv_count, bn_count, fc_count, pooling_count, sum_activation, weighted_sum_neurones, fc_multiply, sum_params_conv, sum_params_bn, sum_params_fc, flatten, global_avg_pooling])
