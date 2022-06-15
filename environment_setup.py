import os
from configparser import ConfigParser

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class ConfigReaderSingleton(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigReaderSingleton, cls).__new__(cls)
            # Put any initialization here. Do not use __init__ for this class.
            # __init__ is called irrespective of whether a new object is getting created or not.
            config_path = os.path.join(PROJECT_ROOT_DIR, 'config.ini')
            cls.instance.parser = ConfigParser()
            cls.instance.parser.read(config_path)
        return cls.instance

    def get_instance(self):
        return self.parser


parser = ConfigReaderSingleton().get_instance()


# Each separate run should have its own config.ini that is stored in a separate folder

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
    list_elements = comma_separated_list.split(",")
    return [x.strip() for x in list_elements]


def write_configs_to_disk():
    base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
    filename = os.path.join(base_log_dir, "configs_for_run.cfg")
    with open(filename, 'w') as configfile:
        parser.write(configfile)
