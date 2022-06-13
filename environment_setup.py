import os
from configparser import ConfigParser

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(PROJECT_ROOT_DIR, 'config.ini')
parser = ConfigParser()
parser.read(config_path)


def get_configurations_dtype_string(section, key, default_value=None):
    return parser[section].get(key, fallback=default_value)


def get_configurations_dtype_int(section, key, default_value=None):
    return parser[section].getint(key, fallback=default_value)


def get_configurations_dtype_float(section, key, default_value=None):
    return parser[section].getfloat(key, fallback=default_value)


def get_configurations_dtype_boolean(section, key, default_value=None):
    return parser[section].getboolean(key, fallback=default_value)


def get_configurations_dtype_string_list(section, key, default_value=None):
    comma_separated_list = parser[section].get(key, fallback=default_value)
    return comma_separated_list.split(",")
