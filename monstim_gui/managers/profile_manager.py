import os
import glob
import yaml
from monstim_gui.io.config_repository import ConfigRepository
import logging

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "../../docs/analysis_profiles")
PROFILE_DIR = os.path.abspath(PROFILE_DIR)


class ProfileManager:
    """Handles loading, saving, and listing analysis profiles."""

    def __init__(self, profile_dir=PROFILE_DIR, reference_config=None):
        self.profile_dir = profile_dir
        os.makedirs(self.profile_dir, exist_ok=True)
        self.reference_config = reference_config

    def list_profiles(self):
        files = glob.glob(os.path.join(self.profile_dir, "*.yml"))
        profiles = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp)
                if data and self.reference_config:
                    # Coerce types for the whole profile using reference config
                    data = ConfigRepository.coerce_types(data, self.reference_config)
                try:
                    profiles.append((data.get("name", os.path.splitext(os.path.basename(f))[0]), f, data))  # type: ignore
                except AttributeError as e:
                    logging.error(f"Error loading profile {f}: {e}")
                    continue
        return profiles

    def load_profile(self, filename):
        with open(filename, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
            if data and self.reference_config:
                data = ConfigRepository.coerce_types(data, self.reference_config)
            return data

    def save_profile(self, data, filename=None):
        if not filename:
            name = data.get("name", "profile")
            filename = os.path.join(
                self.profile_dir, f"{name.replace(' ', '_').lower()}.yml"
            )
        # Use a regular dict for YAML dumping (insertion order is preserved in Python 3.7+)
        ordered = {}
        for key in [
            "name",
            "description",
            "latency_window_preset",
            "stimuli_to_plot",
            "analysis_parameters",
        ]:
            if key in data:
                ordered[key] = data[key]
        # Add any extra keys at the end
        for k, v in data.items():
            if k not in ordered:
                ordered[k] = v
        with open(filename, "w", encoding="utf-8") as fp:
            yaml.safe_dump(ordered, fp, sort_keys=False)
        return filename

    def delete_profile(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
