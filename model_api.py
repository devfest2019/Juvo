# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 01:26:59 2019

@author: Justin Won
"""
import sys
import model_helper as mh
import pandas as pd
import model as md


def determine_acceptance(input_dict):
    '''
    Take input form from stdin and print binary decision to stdout.
    '''
    
    # Parse input
    #input_dict = parse_input(argv[1:])
    
    # Preprocess
    preprocessed_data = md.predict(input_dict)
    
    # Load model and predict
    model = mh.load_model()
    decision = model.predict(preprocessed_data)[0] >= 0.5
    
    # Output to stdout
    return decision
    
def parse_input(argv):
    keys = [key for i, key in enumerate(argv) if i%2 == 0]
    values = []
    for i, value in enumerate(argv):
        if i%2 == 1:
            try:
                values.append(int(value))
            except Exception:
                values.append(bool(value))
    print(type(values[-1]))
    return dict(zip(keys, values))
if __name__ == '__main__':
    # determine_acceptance(sys.argv)
    
    # print(parse_input(['0', '1', '2', '3', '4', 'True']))
    
