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
        return os.path.join(config_dir, "config-user.yml")

    def read_config(self) -> dict:
        with open(self.default_config_file, "r") as file:
            config = yaml.safe_load(file)
        if os.path.exists(self.user_config_file):
            with open(self.user_config_file, "r") as file:
                user_config = yaml.safe_load(file)
            if user_config:
                # Coerce user config types to match default config
                coerced_user = self.coerce_types(user_config, config)
                self._update_nested_dict(config, coerced_user)
        return config

    def write_config(self, config: dict) -> None:
        with open(self.user_config_file, "w") as file:
            yaml.safe_dump(config, file)

    def _update_nested_dict(self, d: dict, u: dict) -> dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    @staticmethod
    def coerce_types(user_data, reference_data):
        """
        Recursively coerce user_data values to the types in reference_data.
        Handles dicts, lists, int, float, bool, etc.
        """
        import ast

        if isinstance(reference_data, dict) and isinstance(user_data, dict):
            result = {}
            for k, v in user_data.items():
                ref_v = reference_data.get(k)
                result[k] = ConfigRepository.coerce_types(v, ref_v)
            return result
        elif isinstance(reference_data, list) and isinstance(user_data, list):
            if reference_data:
                return [ConfigRepository.coerce_types(v, reference_data[0]) for v in user_data]
            else:
                return user_data
        elif isinstance(reference_data, bool):
            if isinstance(user_data, bool):
                return user_data
            if isinstance(user_data, str):
                return user_data.lower() in ("1", "true", "yes", "on")
            return bool(user_data)
        elif isinstance(reference_data, int):
            try:
                return int(user_data)
            except Exception:
                return user_data
        elif isinstance(reference_data, float):
            try:
                return float(user_data)
            except Exception:
                return user_data
        elif isinstance(reference_data, str):
            return str(user_data)
        # Try to parse string as literal if reference is None
        if isinstance(user_data, str):
            try:
                return ast.literal_eval(user_data)
            except Exception:
                return user_data
        return user_data
