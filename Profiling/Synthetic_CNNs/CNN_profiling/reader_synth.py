import json
import csv

if __name__ == "__main__" :

    idxs = []

    models = []
    models_l = []
    input_sizes = [32, 56, 64, 75, 90, 100, 112, 128, 150, 200, 224, 240, 256, 300, 320, 331, 350, 400, 
    448, 480, 500, 512, 568, 600, 620, 700, 720, 750, 800, 820, 850, 900]


    
    #Read the csv file
    with open('./generated_models_12.csv', newline='') as csvfile :
        csv_reader = csv.reader(csvfile)
        for idx, row in enumerate(csv_reader, start=0) :
            print(row)
            if(row[0]!= '' and row[0]!= 'conv_tab' and row[0]!= 'after_conv' and row[0]!= 'top_tab'
            and  int(row[0])) in input_sizes : 
                idxs.append(idx)
                models.append('basic_model_'+str(row[0])+'_'+str(row[1]))
    
    
    """  
    idx = 20
    lines = []

    with open('./generated_models.csv', newline='') as csvfile :
        csv_reader = csv.reader(csvfile)
        csv_data = list(csv_reader)
        #for idx in idxs :
            #models.append('basic_model_'+str(csv_data[idx][0])+'_'+str(csv_data[idx][1]))
        line = csv_data[idx]
        while (idx <= 4338 and line[0] != ''):
            lines.append(line)
            line = csv_data[idx]
            idx = idx+1

    lines.pop(0)
    print(lines)
    #print('basic_model_'+str(lines[0][0])+'_'+str(lines[0][1]))

    """
    
    #Read the csv file
    with open('./models_info_new_12.csv', newline='') as csvfile :
        csv_reader = csv.reader(csvfile)
        for idx, row in enumerate(csv_reader, start=0) :
            models_l.append(row[0])
    

    print(models)
    print("******************************")
    print(models_l)
    print("******************************")
    print(models==models_l)
    print(len(models))
    print("******************************")
    print(idxs)
   

