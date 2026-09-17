"""In-app Markdown help navigation coverage."""

import re
from pathlib import Path

import markdown
import pytest
from markdown.extensions.tables import TableExtension
from PIL import Image
from PySide6.QtCore import QUrl
from PySide6.QtTest import QTest

from monstim_gui.dialogs.help_about import (
    HelpWindow,
    _available_help_image_width,
    _fit_local_help_images,
    _normalise_help_tables,
    _render_tex_to_img,
    _resolve_local_help_images,
)
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


def test_help_repository_resolves_local_assets_and_blocks_escape():
    repository = HelpFileRepository(get_docs_path())

    assert repository.resolve_help_asset("user/using_monstim.md", "../assets/demo/single-recording.png") == (
        Path(get_docs_path()) / "assets" / "demo" / "single-recording.png"
    )
    assert repository.resolve_help_asset("user/using_monstim.md", "../../README.md") is None
    assert repository.resolve_help_asset("user/using_monstim.md", "https://example.test/image.png") is None


def test_home_gallery_assets_resolve_for_the_in_app_help_renderer():
    repository = HelpFileRepository(get_docs_path())

    for asset in (
        "assets/demo/session-emg.png",
        "assets/demo/single-recording.png",
        "assets/demo/reflex-curves.png",
        "assets/demo/recruitment-curves.png",
        "assets/demo/m-max.png",
        "assets/demo/latency-windows.png",
        "assets/demo/vibration-emg.png",
        "assets/demo/stretch-emg.png",
    ):
        assert repository.resolve_help_asset("index.md", asset) == Path(get_docs_path()) / asset


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


def test_help_table_normalisation_sets_qt_friendly_geometry():
    html = """<table>
<thead><tr><th>Method</th><th>Calculation</th><th>Units</th><th>Important limit</th></tr></thead>
<tbody><tr><td>rms</td><td>formula</td><td>signal units</td><td>Limit</td></tr></tbody>
</table>"""

    normalised = _normalise_help_tables(html)

    assert '<table width="100%" border="1" cellspacing="0" cellpadding="6">' in normalised
    assert re.findall(r'<th\b[^>]* width="(\d+%)"[^>]*>', normalised) == ["25%", "30%", "14%", "31%"]
    assert normalised.count('valign="top"') == 8


def test_help_table_normalisation_evenly_allocates_generic_columns():
    html = "<table><thead><tr><th>One</th><th>Two</th><th>Three</th></tr></thead></table>"

    normalised = _normalise_help_tables(html)

    assert re.findall(r'<th\b[^>]* width="(\d+%)"[^>]*>', normalised) == ["34%", "33%", "33%"]


def test_help_window_applies_document_style_and_normalised_tables():
    repository = HelpFileRepository(get_docs_path())
    dialog = HelpWindow(
        repository.read_help_file("science/analysis_methods.md"),
        help_repository=repository,
        source_file="science/analysis_methods.md",
    )

    try:
        assert "th, td" in dialog.text_browser.document().defaultStyleSheet()
        assert '<table width="100%" border="1" cellspacing="0" cellpadding="6">' in dialog._html_template
    finally:
        dialog.close()


def test_help_window_resolves_relative_image_sources_to_local_urls():
    repository = HelpFileRepository(get_docs_path())
    html = '<p><img alt="Demo" src="../assets/demo/single-recording.png" /></p>'

    rendered = _resolve_local_help_images(html, repository, "user/using_monstim.md")

    expected = (Path(get_docs_path()) / "assets" / "demo" / "single-recording.png").resolve().as_uri()
    assert expected in rendered


def test_fit_local_help_images_preserves_native_size_and_aspect_ratio():
    docs_path = Path(get_docs_path())
    image_path = (docs_path / "assets" / "demo" / "single-recording.png").resolve()
    source = image_path.as_uri()
    html = f'<img src="{source}">'

    unconstrained = _fit_local_help_images(html, docs_path, max_width=2_000)
    constrained = _fit_local_help_images(html, docs_path, max_width=400)

    assert unconstrained == html
    assert 'width="400"' in constrained
    with Image.open(image_path) as image:
        expected_height = round(image.height * 400 / image.width)
    assert f'height="{expected_height}"' in constrained


def test_available_help_image_width_reserves_layout_gutter():
    assert _available_help_image_width(500) == 452
    assert _available_help_image_width(30) == 1


def test_fit_local_help_images_leaves_external_and_outside_docs_images_unchanged():
    docs_path = Path(get_docs_path())
    external = '<img src="https://example.test/demo.png">'
    outside = '<img src="file:///C:/outside-docs.png">'

    assert _fit_local_help_images(external, docs_path, max_width=100) == external
    assert _fit_local_help_images(outside, docs_path, max_width=100) == outside


def test_using_monstim_help_window_embeds_demo_images_from_bundled_docs():
    repository = HelpFileRepository(get_docs_path())
    dialog = HelpWindow(
        repository.read_help_file("user/using_monstim.md"),
        help_repository=repository,
        source_file="user/using_monstim.md",
    )

    try:
        expected = (Path(get_docs_path()) / "assets" / "demo" / "single-recording.png").resolve().as_uri()
        assert expected in dialog._html_template
        dialog.resize(450, 550)
        dialog.show()
        QTest.qWait(75)
        expected_width = _available_help_image_width(dialog.text_browser.viewport().width())
        assert f'width="{expected_width}"' in dialog.text_browser.document().toHtml()
        assert dialog.text_browser.horizontalScrollBar().maximum() == 0

        dialog.resize(800, 550)
        QTest.qWait(75)
        resized_width = _available_help_image_width(dialog.text_browser.viewport().width())
        assert resized_width > expected_width
        assert f'width="{resized_width}"' in dialog.text_browser.document().toHtml()
    finally:
        dialog.close()


def test_help_window_resize_preserves_the_current_reading_position():
    repository = HelpFileRepository(get_docs_path())
    dialog = HelpWindow(
        repository.read_help_file("user/using_monstim.md"),
        help_repository=repository,
        source_file="user/using_monstim.md",
    )

    try:
        dialog.resize(450, 300)
        dialog.show()
        QTest.qWait(75)
        scrollbar = dialog.text_browser.verticalScrollBar()
        assert scrollbar.maximum() > 0
        scrollbar.setValue(scrollbar.maximum() // 2)
        reading_position = scrollbar.value()

        dialog.resize(800, 300)
        QTest.qWait(75)

        assert scrollbar.value() == min(reading_position, scrollbar.maximum())
    finally:
        dialog.close()


def test_home_gallery_embeds_each_demo_image_from_bundled_docs():
    repository = HelpFileRepository(get_docs_path())
    dialog = HelpWindow(repository.read_help_file("index.md"), help_repository=repository, source_file="index.md")

    try:
        for asset in (
            "session-emg",
            "single-recording",
            "reflex-curves",
            "recruitment-curves",
            "m-max",
            "latency-windows",
            "vibration-emg",
            "stretch-emg",
        ):
            expected = (Path(get_docs_path()) / "assets" / "demo" / f"{asset}.png").resolve().as_uri()
            assert expected in dialog._html_template
    finally:
        dialog.close()


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
