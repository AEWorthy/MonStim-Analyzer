import os
import yaml

class ConfigRepository:
    """
    Handles reading and writing of configuration files for the GUI.
    """
    def __init__(self, default_config_file: str, user_config_file: str = None):
        self.default_config_file = default_config_file
        self.user_config_file = user_config_file or self._get_user_config_path()

    def _get_user_config_path(self) -> str:
        config_dir = os.path.dirname(self.default_config_file)
        return os.path.join(config_dir, 'config-user.yml')

    def read_config(self) -> dict:
        with open(self.default_config_file, 'r') as file:
            config = yaml.safe_load(file)
        if os.path.exists(self.user_config_file):
            with open(self.user_config_file, 'r') as file:
                user_config = yaml.safe_load(file)
            if user_config:
                self._update_nested_dict(config, user_config)
        return config

    def write_config(self, config: dict) -> None:
        with open(self.user_config_file, 'w') as file:
            yaml.safe_dump(config, file)

    def _update_nested_dict(self, d: dict, u: dict) -> dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
