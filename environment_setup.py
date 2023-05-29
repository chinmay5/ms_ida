import os
import torch
from configparser import ConfigParser

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
default_config_file = 'config_1_y.ini'
new_config_File = input(f"Enter the config file name for execution. Default is {default_config_file} ->").strip()
# new_config_File = ""
# new_config_File = "config_1_y.ini"
config_file = new_config_File if len(new_config_File) > 0 else default_config_file
assert config_file in os.listdir(os.path.join(PROJECT_ROOT_DIR, 'config_files')), f"{config_file} does not exist"

# Check if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConfigReaderSingleton(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigReaderSingleton, cls).__new__(cls)
            # Put any initialization here. Do not use __init__ for this class.
            # __init__ is called irrespective of whether a new object is getting created or not.
            print(f"Using configurations from {os.environ.get('CONFIG_PATH', f'config_files/{config_file}')}")
            config_path = os.environ.get('CONFIG_PATH', os.path.join(PROJECT_ROOT_DIR, f'config_files/{config_file}'))
            cls.instance.parser = ConfigParser()
            cls.instance.parser.read(config_path)

        return cls.instance

    def get_instance(self):
        return self.parser


parser = ConfigReaderSingleton().get_instance()


def get_configurations_dtype_string(section, key):
    return parser[section].get(key)


def get_configurations_dtype_int(section, key):
    return parser[section].getint(key)


def get_configurations_dtype_float(section, key):
    return parser[section].getfloat(key)


def get_configurations_dtype_boolean(section, key):
    return parser[section].getboolean(key)


def get_configurations_dtype_string_list(section, key):
    comma_separated_list = parser[section].get(key)
    list_elements = comma_separated_list.split(",")
    return [x.strip() for x in list_elements]


def get_configurations_dtype_int_list(section, key):
    comma_separated_list = parser[section].get(key)
    list_elements = comma_separated_list.split(",")
    return [int(x.strip()) for x in list_elements]


def write_configs_to_disk():
    """
    Should be called only from the main process.
    Would persist the config file and all readings would be made from it.
    :return:
    """
    base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
    os.makedirs(base_log_dir, exist_ok=True)
    filename = os.path.join(base_log_dir, "configs_for_run.cfg")
    with open(filename, 'w') as configfile:
        parser.write(configfile)
    # Add the path to the subprocess
    os.environ['CONFIG_PATH'] = filename
    # subprocess.run(f"export CONFIG_PATH={filename}")
