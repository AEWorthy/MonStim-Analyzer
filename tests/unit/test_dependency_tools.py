import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_tool(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_deps = load_tool("check_deps", "tools/check_deps.py")
sync_env = load_tool("sync_env_from_requirements", "tools/sync_env_from_requirements.py")


def write_dependency_files(tmp_path, requirements, environment):
    (tmp_path / "requirements.txt").write_text(requirements, encoding="utf-8")
    (tmp_path / "environment.yml").write_text(environment, encoding="utf-8")


def test_check_deps_accepts_pip_matplotlib_for_conda_matplotlib_base(tmp_path, monkeypatch):
    write_dependency_files(
        tmp_path,
        "matplotlib>=3.10\n",
        """
name: test
dependencies:
  - python=3.14.*
  - matplotlib-base=3.10.9
""",
    )

    monkeypatch.chdir(tmp_path)

    assert check_deps.main() == 0


def test_sync_keeps_matplotlib_base_out_of_pip_subsection(tmp_path):
    write_dependency_files(
        tmp_path,
        "matplotlib>=3.10\nopenpyxl>=3.1\n",
        """
name: test
dependencies:
  - python=3.14.*
  - matplotlib-base=3.10.9
  - pip
  - pip:
      - old-package==1.0
""",
    )

    sync_env.sync(
        env_path=tmp_path / "environment.yml",
        req_path=tmp_path / "requirements.txt",
    )

    data = yaml.safe_load((tmp_path / "environment.yml").read_text(encoding="utf-8"))
    deps = data["dependencies"]
    pip_items = next(item["pip"] for item in deps if isinstance(item, dict) and "pip" in item)

    assert "matplotlib-base=3.10.9" in deps
    assert "matplotlib>=3.10" not in pip_items
    assert "openpyxl>=3.1" in pip_items
