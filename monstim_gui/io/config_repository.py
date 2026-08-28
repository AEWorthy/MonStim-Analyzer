import os
from collections.abc import Mapping

import yaml

from monstim_signals.core.configuration import ConfigResolver, ResolvedConfig
from monstim_signals.core.utils import clear_config_cache


class ConfigRepository:
    """
    Handles reading and writing of configuration files for the GUI.
    """

    def __init__(self, default_config_file: str, user_config_file: str | None = None):
        self.default_config_file = default_config_file
        self.user_config_file = user_config_file or self._get_user_config_path()
        self.resolver = ConfigResolver(self.default_config_file, self.user_config_file)

    def _get_user_config_path(self) -> str:
        config_dir = os.path.dirname(self.default_config_file)
        return os.path.join(config_dir, "config-user.yml")

    def read_config(self) -> dict:
        # ``read_config`` is the storage/UI API and may contain application
        # keys or deliberately partial configs.  Domain consumers call
        # ``resolve_config`` when they need the validated typed sections.
        return self.resolver.load_raw()

    def read_default_config(self) -> dict:
        """Return the shipped configuration before any user override is merged."""
        with open(self.default_config_file, encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def resolve_config(self, profile: dict | None = None) -> ResolvedConfig:
        return self.resolver.resolve(profile)

    def refresh(self) -> None:
        self.resolver.invalidate()
        clear_config_cache()

    def write_config(self, config: dict) -> None:
        """Persist only values that differ from the shipped configuration.

        The UI works with the resolved configuration, but writing that complete
        mapping made every shipped default a permanent user override.  Keeping
        the file sparse lets new application defaults take effect naturally.
        """
        config = self._user_overrides(config, self.read_default_config())
        with open(self.user_config_file, "w") as file:
            yaml.safe_dump(config, file)
        self.refresh()

    @classmethod
    def _user_overrides(cls, values, defaults):
        """Return the recursive subset of ``values`` differing from defaults."""
        if isinstance(values, Mapping) and isinstance(defaults, Mapping):
            result = {}
            for key, value in values.items():
                if key not in defaults:
                    result[key] = value
                    continue
                override = cls._user_overrides(value, defaults[key])
                if override != {}:
                    result[key] = override
            return result
        return {} if values == defaults else values

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
