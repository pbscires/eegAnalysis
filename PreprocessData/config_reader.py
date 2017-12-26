'''
Created on Dec 25, 2017

@author: pb8xe
'''

from util.JsonReader import JsonReader

reader = JsonReader("..\\Configuration\\windows_laptop_config.json")
root = reader.get_value("RootDirectory")

subject_directories = reader.get_value("SubjectDirectories")

