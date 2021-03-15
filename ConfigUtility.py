import configparser
import os


class ConfigUtility(object):
    __slots__ = ['_config','root_dir']

    def __init__(self):
        self._config = configparser.RawConfigParser()
        self.root_dir os.path.dirname(os.path.abspath(__file__))


class EnvironmentConfig(ConfigUtility):
    __slots__ = ['config_file_path']

    def __init__(self):
        super().__init__()
        self.config_file_path = os.path.join(self.root_dir,"environment.properties")
        self._config.read(self.config_file_path)

    def load_model(self):
        return self._config.get("Models","model_name")

    def get_training_path(self):
        return self._config.get("Data","training_path")

    def get_valid_path(self):
        return self._config.get("Data","valid_path")

    def get_test_path(self):
        return self._config.get("Data","test_path")

    def get_epochs(self):
        return self._config.get("Model_props","model_name")

    def get_verbose(self):
        return self._config.get("Model_props","verbose")
