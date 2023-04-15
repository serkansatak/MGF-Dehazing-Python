import argparse
import yaml
import os
from Utils.ImageUtils import boxfilter
import numpy as np

def parseParams():
    
    os.getcwd()
    
    parser = argparse.ArgumentParser(description="MGF Dehazing Python Parameters.")
    
    parser.add_argument("--input",  type=str, default="../MGF_dehazing/Data/Ship_input/", help="Input video path or sequence dir that is going to be dehazed.")
    parser.add_argument("--output", default="../MGF_dehazing/Output/Ship_proposed/", type=str, help="Output video path.")
    
    args = parser.parse_args()
    
    with open("./config.yaml", 'r') as f:
        config = yaml.safe_load(f)
            
    for key, value in config.items():
        setattr(args, key, value)
        
    setattr(args, 'N', boxfilter(np.ones((args.height, args.width), dtype=np.float32), args.r))
    setattr(args, 'NN', boxfilter(np.ones((args.height, args.width), dtype=np.float32), args.rr))
        
    return args