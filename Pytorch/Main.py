import os
import argparse
import time
import csv
import torch as torch
from torchvision import transforms
from PIL import Image


# The followin function will create a csv file, which contains the times spent on the predections, recorded for the 30 iterastions
def create_csv(model_name, batch_size, times, output_path):
    with open(output_path, 'w', newline='') as file :
        writer = csv.writer(file)
        writer.writerow([""])
        writer.writerow(["Model name : "+model_name, "Batch_size : "+batch_size])
        writer.writerow(["iteration", "inference time per batch", "inference time per image"])
        for idx, time in enumerate(times, start=0) :
            writer.writerow([idx+1, time*1000, (time/int(batch_size))*1000])


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="MobileNet-v1", type=str, help="Specifiy the name of the model")
    parser.add_argument("image_test", default="elephant", type=str, help="Specifiy the image for predicition")
    parser.add_argument("batch_size", default="1", type=str, help="Specifiy the batch size")
    parser.add_argument("target_device", default="Nano", type=str, help="Specifiy the target device")
    args = parser.parse_args()

    image_size = (240,240,3)

    models_infos = { 
            "DenseNet-121" : {"path": "./Models/DenseNet-121","saved_model" : "DenseNet-121.pt",
            "output_names" : "fc1000/Softmax", "input_names" : "input_1"},
            "DenseNet-169" : {"path": "./Models/DenseNet-169","saved_model" : "DenseNet-169.pt",
            "output_names" : "fc1000/Softmax", "input_names" : "input_1"},
            "DenseNet-201" : {"path": "./Models/DenseNet-201","saved_model" : "DenseNet-201.pt",
            "output_names" : "fc1000/Softmax", "input_names" : "input_1"},
            "Inception-v3" : {"path" : "./Models/Inception-v3", "saved_model" : "Inception-v3.pt",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},
            "MobileNet-v1" : {"path" : "./Models/MobileNet-v1", "saved_model" : "MobileNet-v1.pt",
            "output_names" : "act_softmax/Softmax", "input_names" : "input_1"},
            "MobileNet-v2" :  {"path" : "./Models/MobileNet-v2", "saved_model" : "MobileNet-v2.pt",
            "output_names" : "Logits/Softmax", "input_names" : "input_1"},
            "NASNet-Mobile" : {"path" : "./Models/NASNet-Mobile", "saved_model" : "NASNet-Mobile.pt",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},                                                                                                                                                              
            "NASNet-Large" : {"path" : "./Models/NASNet-Large", "saved_model" : "NASNet-Large.pt",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},
            "ResNet-50" : {"path" : "./Models/ResNet-50", "saved_model" : "ResNet-50.pt",
            "output_names" : "fc1000/Softmax", "input_names" : "input_1"},
            "ResNet-101" : {"path" : "./Models/ResNet-101", "saved_model" : "ResNet-101.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            "ResNet-152" : {"path" : "./Models/ResNet-152", "saved_model" : "ResNet-152.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            "ResNet-50V2" : {"path" : "./Models/ResNet-50V2", "saved_model" : "ResNet-50V2.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            "ResNet-101V2" : {"path" : "./Models/ResNet-101V2", "saved_model" : "ResNet-101V2.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            "ResNet-152V2" : {"path" : "./Models/ResNet-152V2", "saved_model" : "ResNet-152V2.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            "InceptionResNetV2" : {"path" : "./Models/InceptionResNetV2", "saved_model" : "InceptionResNetV2.pt",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},
            "Xception" : {"path" : "./Models/Xception", "saved_model" : "Xception.pt",
            "output_names" : "predictions/Softmax", "input_names" : "input_1"},
            "EfficientNet-B1" : {"path" : "./Models/EfficientNet-B1", "saved_model" : "EfficientNet-B1.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            "EfficientNet-B3" : {"path" : "./Models/EfficientNet-B3", "saved_model" : "EfficientNet-B3.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            "EfficientNet-B5" : {"path" : "./Models/EfficientNet-B5", "saved_model" : "EfficientNet-B5.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"},
            "EfficientNet-B7" : {"path" : "./Models/EfficientNet-B7", "saved_model" : "EfficientNet-B7.pt",
            "output_names" : "probs/Softmax", "input_names" : "input_1"}
            }

    #get the information from the passed args, 
    model_info = models_infos[args.model_name]
    batch_size = int(args.batch_size)
    saved_model_path = model_info['path']+'/Saved-Model/'+model_info['saved_model']

    saved_model = torch.load(saved_model_path)
    saved_model.eval()

    #Prepare data for inference
    transform = transforms.Compose([            
    transforms.Resize(256),                    
    transforms.CenterCrop(224),               
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
    )])

    img = Image.open("./data/elephant.jpg")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with open('./ImageNet_classes/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    times_ = []

    print('start of the ten first predicitions (not counted)')
    for i in range(0,10):
        # Run Inference
        out = saved_model(batch_t)

    print("********* Start of the real measurments (20 predictions) **********")
    for i in range(0,20) :
        #torch.cuda.synchronize()
        start = time.time()
        out = saved_model(batch_t)
        #torch.cuda.synchronize()
        end = time.time()
        #print the top 5 classes predicted by the model
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])
        delta = end - start
        print(start, end, delta)
        times_.append(delta)
    create_csv(args.model_name, args.batch_size, times_, "./csv_files_"+args.target_device+"/"+args.model_name+"_"+str(batch_size)+".csv")
    print(times_)
    #mean_delta = np.array(times_).mean()
    #fps = 1 / mean_delta
    #print('inference time for the batch (ms): ', mean_delta*1000)
    #print('FPs = ', fps)
    print("********* End of the real measurments (20 predictions) **********")