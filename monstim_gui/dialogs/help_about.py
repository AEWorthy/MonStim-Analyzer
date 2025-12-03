import hashlib
import io
import logging
import os
import re
from pathlib import Path

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from matplotlib import pyplot as plt
from mdx_math import MathExtension
from PySide6.QtCore import QEvent, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QFont, QIcon, QImage, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from monstim_gui.core.splash import SPLASH_INFO
from monstim_signals.core import get_source_path

# Cache stores tuples of (path, render_w, render_h, display_w, display_h)
_IMG_CACHE: dict[str, tuple[str, int, int, int, int]] = {}


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

    # Consistent math font
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure(figsize=(0.01, 0.01), dpi=_RENDER_DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Use white text for dark mode, black for light mode
    text_color = "white" if dark_mode else "black"
    ax.text(0.5, 0.5, f"${tex}$", ha="center", va="center", fontsize=fontsize, color=text_color)

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=_RENDER_DPI, transparent=True, bbox_inches="tight", pad_inches=0.01)
    finally:
        plt.close(fig)

    png_bytes = buf.getvalue()
    out_path.write_bytes(png_bytes)

    # Get actual image dimensions (at render DPI)
    img = QImage()
    img.loadFromData(png_bytes)
    render_w, render_h = img.width(), img.height()

    # Calculate display dimensions (scaled down by DPI ratio)
    display_w = int(render_w * _DPI_SCALE)
    display_h = int(render_h * _DPI_SCALE)

    logging.debug(f"math-img render: {out_path.name} ({render_w}x{render_h}px -> display {display_w}x{display_h}px)")
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

    img_path, render_w, render_h, display_w, display_h = _render_tex_to_img(tex, fontsize=render_fontsize, dark_mode=dark_mode)

    # Use proper file:// URI formatting for cross-platform compatibility
    try:
        img_url = Path(img_path).as_uri()
    except Exception:
        # Fallback: ensure forward slashes and basic quoting
        img_url = f'file:///{str(img_path).replace(chr(92), "/")}'

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


def _replace_placeholders_with_images(
    html: str, math_items: list[tuple[str, bool]], scale: float = 1.0, dark_mode: bool = False
) -> str:
    """Replace math placeholders with actual image tags at the given scale."""

    def _sub(m):
        idx = int(m.group(1))
        tex, is_display = math_items[idx]
        return _make_img_tag(tex, is_display, scale, dark_mode)

    return re.sub(r"<!--MATH:(\d+)-->", _sub, html)


class HelpWindow(QDialog):
    """Help window that renders Markdown with LaTeX math as images.

    Supports Ctrl+wheel zoom which scales both text and math images together.
    """

    def __init__(self, markdown_content, title=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title if title else "Help")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), "info.png")))
        self.resize(650, 550)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)

        # Store for zoom re-rendering
        self._markdown_content = markdown_content
        self._zoom_scale = 1.0
        self._pending_zoom_scale = 1.0  # Accumulated zoom before debounce fires
        self._text_zoom_level = 0  # Track text zoom level (0 = default)
        self._pending_text_zoom_delta = 0  # Accumulated text zoom delta
        self._html_template = ""  # HTML with placeholders
        self._math_items: list[tuple[str, bool]] = []
        self._dark_mode = _is_dark_mode()  # Cache dark mode state

        # Debounce timer for zoom - waits for user to stop scrolling
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(50)  # 50ms debounce
        self._zoom_timer.timeout.connect(self._apply_pending_zoom)

        layout = QVBoxLayout(self)

        # Create text browser
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)

        # Install event filter on the viewport (where wheel events actually go)
        self.text_browser.viewport().installEventFilter(self)

        self._initial_render()
        layout.addWidget(self.text_browser)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

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
        self._html_template, self._math_items = _replace_math_with_placeholders(html)

        # Render at current scale
        self._update_html()

    def _update_html(self):
        """Update HTML with math images at current scale."""
        final_html = _replace_placeholders_with_images(
            self._html_template, self._math_items, self._zoom_scale, self._dark_mode
        )

        # Store scroll position (as fraction of total)
        scrollbar = self.text_browser.verticalScrollBar()
        scroll_max = scrollbar.maximum() if scrollbar else 0
        scroll_frac = scrollbar.value() / scroll_max if scroll_max > 0 else 0

        self.text_browser.setHtml(final_html)

        # Restore scroll position (as fraction of new total)
        if scrollbar:
            new_max = scrollbar.maximum()
            scrollbar.setValue(int(scroll_frac * new_max))

    def _update_zoom(self, delta: int):
        """Queue a zoom update (debounced to prevent lag during rapid scrolling)."""
        # Accumulate the zoom scale change
        if delta > 0:
            self._pending_zoom_scale = min(3.0, self._pending_zoom_scale * 1.15)
            self._pending_text_zoom_delta += 1
        else:
            self._pending_zoom_scale = max(0.4, self._pending_zoom_scale / 1.15)
            self._pending_text_zoom_delta -= 1

        # Restart the debounce timer
        self._zoom_timer.start()

    def _apply_pending_zoom(self):
        """Apply the accumulated zoom after debounce delay."""
        if self._pending_zoom_scale != self._zoom_scale:
            self._zoom_scale = self._pending_zoom_scale

            # Apply accumulated text zoom
            if self._pending_text_zoom_delta > 0:
                self.text_browser.zoomIn(self._pending_text_zoom_delta)
            elif self._pending_text_zoom_delta < 0:
                self.text_browser.zoomOut(-self._pending_text_zoom_delta)

            self._text_zoom_level += self._pending_text_zoom_delta
            self._pending_text_zoom_delta = 0

            logging.debug(f"Zoom applied: {self._zoom_scale:.2f}")
            self._update_html()

    def eventFilter(self, watched, event):
        """Intercept Ctrl+wheel events on the text browser viewport."""
        if watched is self.text_browser.viewport():
            if event.type() == QEvent.Type.Wheel:
                modifiers = event.modifiers()
                if modifiers & Qt.KeyboardModifier.ControlModifier:
                    self._update_zoom(event.angleDelta().y())
                    return True  # Consume the event
        return super().eventFilter(watched, event)


def create_help_window(markdown_content, title=None, parent=None):
    """Create a help window that renders Markdown with LaTeX math as images.

    Supports Ctrl+wheel zoom.
    """
    return HelpWindow(markdown_content, title=title, parent=parent)


def clear_math_cache():
    try:
        if _CACHE_DIR.exists():
            for p in list(_CACHE_DIR.glob("mtx_*.png")):
                try:
                    p.unlink()
                except Exception as e:
                    logging.info(f"Failed to remove cache file {p}: {e}")
            logging.info("Cleared math image cache.")
    except Exception as e:
        logging.info(f"Failed to clear math cache: {e}")


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
