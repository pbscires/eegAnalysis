'''
Created on Dec 25, 2017

@author: pb8xe
'''

import json

class JsonReader:
    json_obj = None
    def __init__(self, path):
        with open(path, 'r') as fp:
            self.json_obj = json.load(fp)
    
    def get_value(self, key):
        return self.json_obj[key]

