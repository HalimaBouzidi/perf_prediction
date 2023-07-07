#from nltk.tokenize import sent_tokenize, word_tokenize
from tokenizer import split_into_sentences
import argparse
import csv


def create_csv(start_, fin_, data, is_swap, output_path):
    with open(output_path, 'w', newline='') as file :
        writer = csv.writer(file)
        writer.writerow(["pysical_RAM", "from : "+start_, "to : "+fin_])
        writer.writerow(["idx", "total", "used", "free", "shared", "buff/cache", "available"])
        if(is_swap) :
            for idx, data_ in enumerate(data, start=0) :
                writer.writerow([idx+1, data_[1], data_[2], data_[3], '/', '/', '/'])
        else :
            for idx, data_ in enumerate(data, start=0) :
                writer.writerow([idx+1, data_[1], data_[2], data_[3], data_[4], data_[5], data_[6]])

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", default="free.txt", type=str, help="Specifiy the file name")
    args = parser.parse_args()

    f = open(args.file_name, "r")

    if(f.mode == 'r') :
        
        f1 = f.readlines()
        lines = int(len(f1)/5)

        RAM = []
        Swap = [] 
        mem_ram = []
        mem_swap = []

        for idx in range (0, lines) :
            RAM.append(f1[(5*idx)+3])
            Swap.append(f1[(5*idx)+4])
        

        start = f1[0]
        end = f1[(lines-1)*5]

        for idx in range(0, lines) :
            mem_ram.append(RAM[idx].split())
            mem_swap.append(Swap[idx].split())

        create_csv(start, end, mem_ram, False, './generated-free-csv/mem_ram.csv')
        create_csv(start, end, mem_swap, True, './generated-free-csv/mem_swap.csv')
