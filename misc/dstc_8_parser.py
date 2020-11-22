import codecs
import pathlib
import numpy as np
import json
import os
import pathlib
import numpy as np

def dstc8_dialig_parser(path_folder):
    """
    Parse folder with dialogues in json into np.array utterances, intends, servises.
    """
    text = []
    intends = []
    service = []
    path_train_folder = pathlib.Path(path_folder)
    list_json_files = os.listdir(path_folder)
    list_json_files.remove('schema.json')
    for file_name in list_json_files:
        with open(path_train_folder / file_name, 'r') as read_file:
            data = json.load(read_file)
            for dialog in data:
                for turn in dialog['turns']:
                    for act in turn['frames'][0]['actions']:
                        if act['act'] in ['OFFER_INTENT', 'INFORM_INTENT']:
                            text.append(turn['utterance'])
                            intends.append(act['values'][0])
                            service.append(turn['frames'][0]['service'])
    return np.array(text, dtype='object'), np.array(intends, dtype='object'),
           np.array(service, dtype='object')

    def dstc_schema_parser(path_folder):
    """
    """
    description = []
    intends = []
    service_names = []
    with open(path_folder / 'schema.json', 'r') as read_file:
        data = json.load(read_file)
        for service in data:
            for intent in service['intents']:
                description.append(intent['description'])
                intends.append(intent['name'])
                service_names.append(service['service_name'])
    return np.array(description, dtype='object'), 
           np.array(intends, dtype='object'), 
           np.array(service_names, dtype='object')