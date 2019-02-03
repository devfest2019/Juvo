# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 01:26:59 2019

@author: Justin Won
"""
import sys
import model_helper as mh
import pandas as pd
import model as md

model = None
def determine_acceptance(input_dict):
    global model
    '''
    Take input form from stdin and print binary decision to stdout.
    '''

    # Parse input
    # input_dict = parse_input(argv[1:])
    for key in input_dict.keys():
        try:
            input_dict[key] = int(input_dict[key])
        except:
            input_dict[key] = True if input_dict[key] == u'on' else False

    # Preprocess
    preprocessed_data = md.predict(input_dict)

    # Load model and predict
    if model == None:
        model = mh.load_model()
    dec_val = model.predict(preprocessed_data)[0]
    print("DECISION VALUE: " + str(dec_val))
    decision = dec_val >= 0.5

    # Output to stdout
    return decision

# def parse_input(argv):
#     keys = [key for i, key in enumerate(argv) if i%2 == 0]
#     values = []
#     for i, value in enumerate(argv):
#         if i%2 == 1:
#             try:
#                 values.append(int(value))
#             except Exception:
#                 values.append(bool(value))
#                 print(type(values[-1]))
#                 return dict(zip(keys, values))
#             if __name__ == '__main__':
#                 # determine_acceptance(sys.argv)
