"""In-app Markdown help navigation coverage."""

import re
from pathlib import Path

import markdown
import pytest
from markdown.extensions.tables import TableExtension
from PySide6.QtCore import QUrl

from monstim_gui.dialogs.help_about import HelpWindow, _render_tex_to_img
from monstim_gui.io.help_repository import HelpFileRepository
from monstim_gui.managers.profile_manager import get_bundled_profile_dir
from monstim_signals.core import get_config_path, get_docs_path


def test_documentation_tree_exposes_help_and_configuration_resources():
    docs_path = Path(get_docs_path())

    assert (docs_path / "user" / "index.md").is_file()
    assert (docs_path / "science" / "analysis_methods.md").is_file()
    assert (docs_path / "developer" / "index.md").is_file()
    assert (docs_path / "resources" / "config.yml").is_file()
    assert get_config_path() == str(docs_path / "resources" / "config.yml")
    assert get_bundled_profile_dir() == str(docs_path / "resources" / "analysis_profiles")


def test_help_repository_resolves_relative_docs_links_and_blocks_escape():
    repository = HelpFileRepository(get_docs_path())

    assert repository.resolve_help_link("science/emg_processing.md", "extrema_peak_to_trough.md#overlap") == (
        "science/extrema_peak_to_trough.md",
        "overlap",
    )
    assert repository.resolve_help_link("user/using_monstim.md", "../../README.md") is None
    with pytest.raises(ValueError, match="outside"):
        repository.read_help_file("../README.md")


def test_all_bundled_markdown_links_resolve_to_bundled_topics():
    repository = HelpFileRepository(get_docs_path())
    links = []
    for document in repository.iter_help_files():
        current_file = document.as_posix()
        document_links = re.findall(r"\]\(([^)#]+)(?:#[^)]+)?\)", repository.read_help_file(document))
        links.extend((current_file, link) for link in document_links if link.lower().endswith(".md"))

    assert links
    assert all(repository.resolve_help_link(current_file, link) is not None for current_file, link in links)


def test_analysis_methods_math_table_keeps_four_columns():
    repository = HelpFileRepository(get_docs_path())
    document = repository.read_help_file("science/analysis_methods.md")
    html = markdown.Markdown(extensions=[TableExtension()]).convert(document)

    rows = re.findall(r"<tr>(.*?)</tr>", html, flags=re.DOTALL)
    assert len(rows) == 8
    assert all(len(re.findall(r"<(?:td|th)", row)) == 4 for row in rows)


def test_analysis_methods_absolute_value_math_renders():
    _, width, height, _, _ = _render_tex_to_img(r"\frac{1}{n}\sum_i \left\vert x_i \right\vert", fontsize=12, dark_mode=False)

    assert width > 1
    assert height > 1


def test_help_window_navigates_relative_markdown_link_and_back():
    repository = HelpFileRepository(get_docs_path())
    dialog = HelpWindow(
        repository.read_help_file("science/emg_processing.md"),
        help_repository=repository,
        source_file="science/emg_processing.md",
    )

    dialog._open_link(QUrl("extrema_peak_to_trough.md"))

    assert dialog._source_file == "science/extrema_peak_to_trough.md"
    assert "Exclusive extrema peak-to-trough" in dialog._markdown_content
    assert dialog.back_button.isEnabled()
    dialog._go_back()
    assert dialog._source_file == "science/emg_processing.md"
    dialog.close()
