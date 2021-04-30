import os
from configparser import ConfigParser

configuration = ConfigParser()
configuration.read("config.ini")

def pre_processing():
    if not os.path.exists(configuration['DIRECTORIES']['model']):
        os.mkdir(configuration['DIRECTORIES']['model'])
    if not os.path.exists(configuration['DIRECTORIES']['files']):
        os.mkdir(configuration['DIRECTORIES']['files'])
    if not os.path.exists(configuration['DIRECTORIES']['raw']):
        os.mkdir(configuration['DIRECTORIES']['raw'])
