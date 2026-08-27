import hashlib
import html as html_lib
import io
import logging
import os
import re
from pathlib import Path

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mdx_math import MathExtension
from PIL import Image
from PySide6.QtCore import QEvent, QStandardPaths, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextBrowser, QToolButton, QVBoxLayout, QWidget

from monstim_gui.core.splash import SPLASH_INFO
from monstim_signals.core import get_source_path

# Cache stores tuples of (path, render_w, render_h, display_w, display_h)
_IMG_CACHE: dict[str, tuple[str, int, int, int, int]] = {}

logger = logging.getLogger(__name__)


# Persist math images in a user-specific cache directory
def _get_cache_dir() -> Path:
    cache_location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    cache_dir = Path(cache_location) / "monstim_math_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


_CACHE_DIR = _get_cache_dir()


# Render DPI - higher = sharper images
_RENDER_DPI = 300


# Reference DPI for sizing (images rendered at _RENDER_DPI will be scaled to display as if at this DPI)
def _get_display_dpi() -> int:
    """Detect the system's display DPI, fallback to 100 if unavailable."""
    app = QApplication.instance()
    screen = app.primaryScreen() if app else None
    if screen is not None:
        dpi = int(screen.logicalDotsPerInch())
        # Clamp to reasonable range
        return max(72, min(600, dpi))
    return 100


_DISPLAY_DPI = _get_display_dpi()
# Scale factor to convert from render size to display size
_DPI_SCALE = _DISPLAY_DPI / _RENDER_DPI


def _is_dark_mode() -> bool:
    """Detect if the application is in dark mode based on window background color."""
    app = QApplication.instance()
    if app:
        palette = app.palette()
        bg_color = palette.color(QPalette.ColorRole.Window)
        # Consider dark mode if background luminance is low
        # Using simple luminance formula: 0.299*R + 0.587*G + 0.114*B
        luminance = 0.299 * bg_color.red() + 0.587 * bg_color.green() + 0.114 * bg_color.blue()
        return luminance < 128
    return False


def _render_tex_to_img(tex: str, fontsize: int = 12, dark_mode: bool = False) -> tuple[str, int, int, int, int]:
    """Render TeX to a high-DPI PNG and return (path, render_w, render_h, display_w, display_h).

    Renders at high DPI for quality, but returns display dimensions scaled down
    so the visual size matches what you'd get at _DISPLAY_DPI.

    Args:
        tex: LaTeX math string
        fontsize: Font size for rendering
        dark_mode: If True, render in white color for dark backgrounds
    """
    key = f"{tex}|{fontsize}|{_RENDER_DPI}|{'dark' if dark_mode else 'light'}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    out_path = _CACHE_DIR / f"mtx_{h}.png"

    if key in _IMG_CACHE and out_path.exists():
        return _IMG_CACHE[key]

    buf = io.BytesIO()
    png_bytes = b""
    fig = None
    try:
        with rc_context({"mathtext.fontset": "stix", "font.family": "DejaVu Sans", "font.size": fontsize}):
            fig = Figure(figsize=(0.01, 0.01), dpi=_RENDER_DPI)
            canvas = FigureCanvasAgg(fig)
            fig.patch.set_alpha(0)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            # Use white text for dark mode, black for light mode
            text_color = "white" if dark_mode else "black"
            ax.text(0.5, 0.5, f"${tex}$", ha="center", va="center", fontsize=fontsize, color=text_color)

            fig.savefig(buf, format="png", dpi=_RENDER_DPI, transparent=True, bbox_inches="tight", pad_inches=0.01)
            png_bytes = buf.getvalue()
            out_path.write_bytes(png_bytes)
    except Exception:
        # If matplotlib or fonts fail in frozen app, log and fall back.
        logger.exception("Failed to save math image via matplotlib")
        try:
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(out_path, format="PNG")
            if out_path.exists():
                logger.debug(f"Wrote fallback transparent image to {out_path}")
                png_bytes = out_path.read_bytes()
            else:
                logger.error("Failed to save fallback transparent PNG.")
        except Exception:
            logger.exception("Fallback transparent PNG save also failed")
    finally:
        if fig is not None:
            try:
                fig.clear()
                del canvas
                del fig
            except TypeError:
                logger.exception("Failed to clear matplotlib figure during cleanup")

    # Read dimensions without Qt: this helper also runs in headless tests
    # before a QApplication exists. QImage initialization/loading may query a
    # screen's DPI and crash on Linux when no GUI application is present.
    render_w, render_h = 0, 0
    try:
        if png_bytes:
            with Image.open(io.BytesIO(png_bytes)) as img:
                render_w, render_h = img.size
        else:
            with Image.open(out_path) as img:
                render_w, render_h = img.size
    except Exception:
        logger.exception(f"Failed to read generated math PNG dimensions for tex='{tex}' from data or file {out_path}")

    if not (render_w and render_h):
        tex_display = f"{tex[:40]}..." if len(tex) > 40 else tex
        logger.error(f"Failed to read generated math PNG dimensions for tex='{tex_display}' from data or file {out_path}")

    # Calculate display dimensions (scaled down by DPI ratio)
    display_w = int(render_w * _DPI_SCALE)
    display_h = int(render_h * _DPI_SCALE)

    result = (str(out_path), render_w, render_h, display_w, display_h)
    _IMG_CACHE[key] = result
    return result


def _make_img_tag(tex: str, is_display: bool, scale: float = 1.0, dark_mode: bool = False) -> str:
    """Create an <img> tag for math with proper pixel sizing.

    Args:
        tex: The LaTeX content
        is_display: True for display math (centered), False for inline
        scale: Zoom scale factor (1.0 = 100%)
        dark_mode: If True, render in white color for dark backgrounds
    """
    # Base fontsizes that look good at scale=1.0
    base_fontsize = 14 if not is_display else 18
    render_fontsize = int(base_fontsize * scale)
    render_fontsize = max(8, min(72, render_fontsize))  # Clamp to reasonable range

    img_path, _, _, display_w, display_h = _render_tex_to_img(tex, fontsize=render_fontsize, dark_mode=dark_mode)

    # Use proper file:// URI formatting for cross-platform compatibility
    # Use QUrl.fromLocalFile for reliable file:// URIs (works with frozen apps)
    try:
        img_url = QUrl.fromLocalFile(str(img_path)).toString()
    except Exception:
        logger.exception(f"Failed to create QUrl for math image {img_path}, falling back to manual URI")
        try:
            img_url = Path(img_path).resolve().as_uri()
        except Exception:
            logger.exception(f"Failed to create URI for math image {img_path}, using raw path")
            img_url = f"file:///{str(img_path).replace(chr(92), '/')}"

    # Use the display dimensions (scaled down from high-DPI render)
    if is_display:
        return f'<div align="center"><img src="{img_url}" width="{display_w}" height="{display_h}"/></div>'
    else:
        return f'<img src="{img_url}" width="{display_w}" height="{display_h}" align="middle"/>'


def _replace_math_with_placeholders(html: str) -> tuple[str, list[tuple[str, bool]]]:
    """Replace math with placeholders and return list of (tex, is_display) tuples."""
    math_items: list[tuple[str, bool]] = []

    def _sub_script(m):
        mode = m.group("mode") or ""
        content = m.group("content")
        is_display = "display" in mode
        idx = len(math_items)
        math_items.append((content, is_display))
        return f"<!--MATH:{idx}-->"

    html = re.sub(
        r"<script\s+type=[\'\"]math/tex(?:;\s*mode=(?P<mode>display))?[\'\"]>(?P<content>.*?)</script>",
        _sub_script,
        html,
        flags=re.DOTALL,
    )

    def _sub_display(m):
        content = m.group("content")
        idx = len(math_items)
        math_items.append((content, True))
        return f"<!--MATH:{idx}-->"

    html = re.sub(r"\$\$(?P<content>.*?)\$\$", _sub_display, html, flags=re.DOTALL)

    def _sub_inline(m):
        content = m.group("content")
        idx = len(math_items)
        math_items.append((content, False))
        return f"<!--MATH:{idx}-->"

    html = re.sub(r"(?<!\$)\$(?P<content>[^$]+)\$(?!\$)", _sub_inline, html, flags=re.DOTALL)

    return html, math_items


def _replace_placeholders_with_images(html: str, math_items: list[tuple[str, bool]], scale: float = 1.0, dark_mode: bool = False) -> str:
    """Replace math placeholders with actual image tags at the given scale."""

    def _sub(m):
        idx = int(m.group(1))
        tex, is_display = math_items[idx]
        return _make_img_tag(tex, is_display, scale, dark_mode)

    return re.sub(r"<!--MATH:(\d+)-->", _sub, html)


def _help_document_stylesheet(dark_mode: bool) -> str:
    """Return a conservative stylesheet supported by Qt rich text.

    QTextBrowser implements a deliberately small CSS subset, so table geometry
    is supplied as HTML attributes in :func:`_normalise_help_tables`.  This
    stylesheet handles the visual details that Qt does support consistently.
    """
    palette = QApplication.palette()
    text = palette.color(QPalette.ColorRole.Text).name()
    background = palette.color(QPalette.ColorRole.Base).name()
    link = palette.color(QPalette.ColorRole.Link).name()
    header_background = "#3c3c3c" if dark_mode else "#e9edf2"
    rule = "#666666" if dark_mode else "#b8c0ca"
    code_background = "#383838" if dark_mode else "#f2f4f6"

    return f"""
        body {{ color: {text}; background-color: {background}; line-height: 1.35; }}
        h1 {{ margin-top: 0; margin-bottom: 14px; }}
        h2 {{ margin-top: 20px; margin-bottom: 8px; }}
        h3 {{ margin-top: 16px; margin-bottom: 6px; }}
        p {{ margin-top: 0; margin-bottom: 10px; }}
        ul, ol {{ margin-top: 3px; margin-bottom: 10px; }}
        li {{ margin-top: 2px; margin-bottom: 2px; }}
        a {{ color: {link}; text-decoration: underline; }}
        code {{ background-color: {code_background}; padding: 1px 3px; }}
        table {{ border: 1px solid {rule}; margin-top: 8px; margin-bottom: 12px; }}
        th {{ background-color: {header_background}; font-weight: bold; }}
        th, td {{ padding: 6px; border: 1px solid {rule}; }}
    """


def _normalise_help_tables(html: str) -> str:
    """Add Qt-friendly geometry and cell attributes to Markdown tables.

    Python-Markdown emits bare table tags.  Qt's rich-text layout otherwise
    sizes those tables from their widest cell, which makes prose columns and
    inline math compete for space.  Attributes are more reliably honoured by
    QTextDocument than modern table CSS.
    """

    def replace_table(match: re.Match[str]) -> str:
        table = match.group(0)
        headers = re.findall(r"<th\b[^>]*>(.*?)</th>", table, flags=re.DOTALL | re.IGNORECASE)
        column_count = len(headers)
        if not column_count:
            return table

        header_names = [html_lib.unescape(re.sub(r"<[^>]+>", "", header)).strip().casefold() for header in headers]
        # Documentation comparison tables often put a compact identifier next
        # to a formula and explanatory prose.  Give each of those a useful
        # minimum share rather than allowing Qt to infer it from a single row.
        if header_names == ["method", "calculation", "units", "important limit"]:
            widths = ["25%", "30%", "14%", "31%"]
        else:
            base_width, remainder = divmod(100, column_count)
            widths = [f"{base_width + (index < remainder)}%" for index in range(column_count)]

        cell_index = 0

        def replace_cell(cell_match: re.Match[str]) -> str:
            nonlocal cell_index
            tag = cell_match.group(1).lower()
            attributes = cell_match.group(2)
            width = widths[cell_index % column_count]
            cell_index += 1
            alignment = ' align="left"' if tag == "th" else ""
            return f'<{tag}{attributes} width="{width}" valign="top"{alignment}>'

        table = re.sub(r"<(th|td)\b([^>]*)>", replace_cell, table, flags=re.IGNORECASE)
        return re.sub(
            r"<table>",
            '<table width="100%" border="1" cellspacing="0" cellpadding="6">',
            table,
            count=1,
            flags=re.IGNORECASE,
        )

    return re.sub(r"<table>.*?</table>", replace_table, html, flags=re.DOTALL | re.IGNORECASE)


class HelpWindow(QDialog):
    """Help window that renders Markdown with LaTeX math as images.

    Supports Ctrl+wheel zoom which scales both text and math images together.
    """

    def __init__(self, markdown_content, title=None, parent=None, *, help_repository=None, source_file: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(title if title else "Help")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), "info.png")))
        self.resize(650, 550)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)

        # Store for zoom re-rendering
        self._markdown_content = markdown_content
        self._help_repository = help_repository
        self._source_file = source_file
        self._history: list[tuple[str, str | None]] = []
        self._pending_anchor = ""
        self._reset_scroll = False
        self._zoom_scale = 1.0
        # Accumulate discrete zoom steps during rapid scrolling (debounced)
        # Positive = zoom in steps, Negative = zoom out steps
        self._pending_zoom_steps = 0
        self._text_zoom_level = 0  # Track text zoom level (0 = default)
        self._pending_text_zoom_delta = 0  # Accumulated text zoom delta
        self._html_template = ""  # HTML with placeholders
        self._math_items: list[tuple[str, bool]] = []
        self._dark_mode = _is_dark_mode()  # Cache dark mode state

        # listen for application palette changes so we can update math images
        # if the user toggles system theme while the help window is open.
        # Use the `paletteChanged` signal when available instead of installing
        # an application event filter (avoids QObject lifetime issues).
        app = QApplication.instance()
        self._app_palette_connected = False
        if app and hasattr(app, "paletteChanged"):
            try:
                app.paletteChanged.connect(self._on_app_palette_changed)
                self._app_palette_connected = True
            except Exception:
                self._app_palette_connected = False
                logger.exception("Failed to connect paletteChanged signal on HelpWindow.")

        # Debounce timer for zoom - waits for user to stop scrolling
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(50)  # 50ms debounce
        self._zoom_timer.timeout.connect(self._apply_pending_zoom)

        layout = QVBoxLayout(self)

        # Create text browser
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenLinks(False)
        self.text_browser.setOpenExternalLinks(False)
        self.text_browser.anchorClicked.connect(self._open_link)

        # Install event filter on the viewport (where wheel events actually go)
        self.text_browser.viewport().installEventFilter(self)

        self._initial_render()
        layout.addWidget(self.text_browser)

        # Close button
        btn_row = QHBoxLayout()
        self.back_button = QToolButton()
        self.back_button.setText("Back")
        self.back_button.setToolTip("Return to the previous help topic")
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(self._go_back)
        btn_row.addWidget(self.back_button)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _open_link(self, url: QUrl) -> None:
        """Open bundled Markdown links in this help window, not a blank browser page."""
        href = url.toString()
        if self._help_repository is not None and self._source_file is not None:
            resolved = self._help_repository.resolve_help_link(self._source_file, href)
            if resolved is not None:
                target_file, anchor = resolved
                self._history.append((self._markdown_content, self._source_file))
                self.back_button.setEnabled(True)
                self._load_document(self._help_repository.read_help_file(target_file), target_file, anchor)
                return
        if url.scheme() in {"http", "https", "mailto"}:
            QDesktopServices.openUrl(url)
            return
        QMessageBox.warning(self, "Help link unavailable", f"The linked help page could not be found:\n{href}")

    def _load_document(self, markdown_content: str, source_file: str, anchor: str = "") -> None:
        self._markdown_content = markdown_content
        self._source_file = source_file
        self._pending_anchor = anchor
        self._reset_scroll = True
        self.setWindowTitle(f"Help — {Path(source_file).stem.replace('_', ' ').title()}")
        self._initial_render()

    def _go_back(self) -> None:
        if not self._history:
            return
        markdown_content, source_file = self._history.pop()
        self.back_button.setEnabled(bool(self._history))
        self._load_document(markdown_content, source_file or "")

    def _initial_render(self):
        """Initial render of markdown content (called once on init)."""
        # Convert markdown and get math placeholders
        md = markdown.Markdown(
            extensions=[
                TableExtension(),
                FencedCodeExtension(),
                CodeHiliteExtension(guess_lang=False),
                MathExtension(enable_dollar_delimiter=True),
            ]
        )
        html = md.convert(self._markdown_content)

        # Extract math and replace with placeholders (only done once)
        html = _normalise_help_tables(html)
        self._html_template, self._math_items = _replace_math_with_placeholders(html)

        # Render at current scale
        self._update_html()

    def _update_html(self):
        """Update HTML with math images at current scale."""
        final_html = _replace_placeholders_with_images(self._html_template, self._math_items, self._zoom_scale, self._dark_mode)

        # Store scroll position (as fraction of total)
        scrollbar = self.text_browser.verticalScrollBar()
        scroll_max = scrollbar.maximum() if scrollbar else 0
        scroll_frac = scrollbar.value() / scroll_max if scroll_max > 0 else 0

        self.text_browser.document().setDefaultStyleSheet(_help_document_stylesheet(self._dark_mode))
        self.text_browser.setHtml(final_html)

        # Restore scroll position (as fraction of new total), unless navigating.
        if scrollbar:
            new_max = scrollbar.maximum()
            scrollbar.setValue(0 if self._reset_scroll else int(scroll_frac * new_max))
        if self._pending_anchor:
            anchor = self._pending_anchor
            self._pending_anchor = ""
            QTimer.singleShot(0, lambda: self.text_browser.scrollToAnchor(anchor))
        self._reset_scroll = False

    def _update_zoom(self, delta: int):
        """Queue a zoom update (debounced to prevent lag during rapid scrolling).

        Instead of multiplying an accumulated scale (which compounds on
        rapid successive wheel events), accumulate discrete steps. When the
        debounce timer fires we apply 1.15 ** steps to the current scale,
        producing predictable behaviour regardless of scroll speed.
        """
        # Each wheel step corresponds to one discrete zoom step
        if delta > 0:
            self._pending_zoom_steps += 1
            self._pending_text_zoom_delta += 1
        else:
            self._pending_zoom_steps -= 1
            self._pending_text_zoom_delta -= 1

        # Restart the debounce timer
        self._zoom_timer.start()

    def _apply_pending_zoom(self):
        """Apply the accumulated zoom after debounce delay."""
        if self._pending_zoom_steps != 0:
            # Compute new scale as current scale multiplied by 1.15^steps
            new_scale = self._zoom_scale * (1.15**self._pending_zoom_steps)
            # Clamp to allowed range
            self._zoom_scale = max(0.4, min(3.0, new_scale))

            # Apply accumulated text zoom
            if self._pending_text_zoom_delta > 0:
                self.text_browser.zoomIn(self._pending_text_zoom_delta)
            elif self._pending_text_zoom_delta < 0:
                self.text_browser.zoomOut(-self._pending_text_zoom_delta)

            self._text_zoom_level += self._pending_text_zoom_delta
            self._pending_text_zoom_delta = 0

            # Reset pending step counter
            self._pending_zoom_steps = 0

            logger.debug(f"Zoom applied: {self._zoom_scale:.2f}")
            self._update_html()

    def eventFilter(self, watched, event):
        """Intercept Ctrl+wheel events on the text browser viewport."""
        # Note: application palette changes are handled via the
        # `paletteChanged` signal when available. Keep this method focused on
        # intercepting Ctrl+wheel on the text browser viewport.

        # Then handle Ctrl+wheel for zooming inside the text browser viewport.
        if watched is self.text_browser.viewport() and event.type() == QEvent.Type.Wheel:
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                self._update_zoom(event.angleDelta().y())
                return True  # Consume the event

        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        """Cleanup any installed application event filter on close."""
        app = QApplication.instance()
        if app and getattr(self, "_app_palette_connected", False):
            try:
                app.paletteChanged.disconnect(self._on_app_palette_changed)
            except Exception:
                logger.exception("Failed to disconnect paletteChanged signal on HelpWindow close.")
        return super().closeEvent(event)

    def _on_app_palette_changed(self):
        """Slot called when the application palette changes."""
        new_dark = _is_dark_mode()
        if new_dark != self._dark_mode:
            self._dark_mode = new_dark
            logger.debug(f"Palette changed (signal), dark_mode={self._dark_mode}")
            self._update_html()


def create_help_window(markdown_content, title=None, parent=None, *, help_repository=None, source_file: str | None = None):
    """Create a help window that renders Markdown with LaTeX math as images.

    Supports Ctrl+wheel zoom.
    """
    return HelpWindow(markdown_content, title=title, parent=parent, help_repository=help_repository, source_file=source_file)


def clear_math_cache():
    try:
        if _CACHE_DIR.exists():
            for p in list(_CACHE_DIR.glob("mtx_*.png")):
                try:
                    p.unlink()
                except Exception:
                    logger.exception(f"Failed to remove cache file {p}")
            logger.info("Cleared math image cache.")
    except Exception:
        logger.exception("Failed to clear math cache.")


class AboutDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Program Information")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), "icon.png")))
        self.setFixedSize(400, 400)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Dialog)

        # Set white background
        self.setStyleSheet("background-color: white;")

        layout = QVBoxLayout(self)

        # Add logo
        logo_pixmap = QPixmap(os.path.join(get_source_path(), "logo.png"))
        max_width = 200
        max_height = 200
        logo_pixmap = logo_pixmap.scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        logo_label = QLabel()
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        font = QFont()
        font.setPointSize(12)

        program_name = QLabel(SPLASH_INFO["program_name"])
        program_name.setStyleSheet("font-weight: bold; color: #333333;")
        program_name.setFont(font)
        program_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(program_name)

        version = QLabel(SPLASH_INFO["version"])
        version.setStyleSheet("color: #666666;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        description = QLabel(SPLASH_INFO["description"])
        description.setStyleSheet("color: #666666;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        copyright = QLabel(SPLASH_INFO["copyright"])
        copyright.setStyleSheet("color: #999999;")
        copyright.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(copyright)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        self.close()
