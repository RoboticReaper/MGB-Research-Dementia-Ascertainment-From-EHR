import os
import argparse
import sys
#import logger_provider
from time import time
from datetime import datetime

class file_seperator():
    def __init__(self):
        self.f_input = ''
        self.output_dir = ''
        self.no_lines_per_file = 0

    def seperator(self, input_file, output_dir, no_lines_per_file):
        self.f_input = input_file
        self.output_dir = output_dir
        self.no_lines_per_file = no_lines_per_file
        output_files = []
        if no_lines_per_file != 0:
            f_index = 0
            output_file = os.path.join(output_dir, 'seperated_file_{}.txt'.format(f_index))
            output_files.append(output_file)

            w = open(output_file, 'wb') 
            # open the file once
            with open(input_file, 'rb') as f:
                for i, line in enumerate(f):
                    # to see if it is necessary to create a new file. 
                    if i%no_lines_per_file == 0:
                        f_index +=1
                        output_file = os.path.join(output_dir, 'seperated_file_{}.txt'.format(f_index))
                        output_files.append(output_file)
                        w.close()
                        # reopen the a new file
                        w = open(output_file, 'wb') 

                    w.write(line)
                    w.flush()
        return output_files


def main():

    parser = argparse.ArgumentParser(description='Seperate large note file into multiple files')
    parser.add_argument("input_file", type=str )
    parser.add_argument("output_dir", default=os.path.join(os.getcwd(), "output"), type=str)
    parser.add_argument("no_lines", default = 0, type=int)
#    parser.add_argument("--no_files", default=0, type=int)    
#    parser.add_argument("--logging-interval", type=int, default=100)

    args = parser.parse_args()
    print(args.input_file)
    print(args.output_dir)
    
    #logging_interval = args.logging_interval
    if args.no_lines != 0:
        f_index = 0
        output_file = os.path.join(args.output_dir, 'seperated_file_{}.txt'.format(f_index))
        w = open(output_file, 'wb') 
        # open the file once
        with open(args.input_file, 'rb') as f:
            for i, line in enumerate(f):
                # to see if it is necessary to create a new file. 
                if i%args.no_lines == 0:
                    f_index +=1
                    output_file = os.path.join(args.output_dir, 'seperated_file_{}.txt'.format(f_index))
                    w.close()
                    # reopen the a new file
                    w = open(output_file, 'wb') 

                w.write(line)
                w.flush()


if __name__=="__main__":
    main()
