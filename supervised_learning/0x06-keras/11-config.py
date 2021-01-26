#!/usr/bin/env python3
"""saves and loads model’s configuration
   in JSON format"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves model in JSON format"""
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)
    return None


def load_config(filename):
    """loads model in JSON format"""
    with open(filename, 'r') as json_file:
        load_model_json = json_file.read()
        return K.models.model_from_json(load_model_json)
