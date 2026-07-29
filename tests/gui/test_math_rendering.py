import os

from matplotlib.figure import Figure

from monstim_gui.dialogs import help_about as ha


def setup_function():
    # Ensure cache and files are cleared before each test
    ha._IMG_CACHE.clear()
    ha.clear_math_cache()


def test_render_and_cache(tmp_path):
    # Render a simple expression and ensure file is created and cached
    res1 = ha._render_tex_to_img("x^2", fontsize=12, dark_mode=False)
    path1 = res1[0]
    assert os.path.exists(path1)

    # Calling again should return the same result (from cache)
    res2 = ha._render_tex_to_img("x^2", fontsize=12, dark_mode=False)
    assert res1 == res2


def test_placeholder_regex_extraction():
    # Script tag
    html = "start <script type='math/tex'>s+1</script> end"
    out, items = ha._replace_math_with_placeholders(html)
    assert len(items) == 1
    assert items[0][0].strip() == "s+1"
    assert "<!--MATH:0-->" in out

    # Display math using $$
    html2 = "before $$E=mc^2$$ after"
    out2, items2 = ha._replace_math_with_placeholders(html2)
    assert len(items2) == 1
    assert items2[0] == ("E=mc^2", True)

    # Inline math using single $
    html3 = "a $b+c$ d"
    out3, items3 = ha._replace_math_with_placeholders(html3)
    assert len(items3) == 1
    assert items3[0] == ("b+c", False)


def test_dark_mode_produces_different_images():
    # Render in light mode
    ha._IMG_CACHE.clear()
    light = ha._render_tex_to_img("x^2", fontsize=12, dark_mode=False)
    # Clear cache to force new render
    ha._IMG_CACHE.clear()
    dark = ha._render_tex_to_img("x^2", fontsize=12, dark_mode=True)

    assert os.path.exists(light[0])
    assert os.path.exists(dark[0])

    with open(light[0], "rb") as f:
        b1 = f.read()
    with open(dark[0], "rb") as f:
        b2 = f.read()
    # The bytes should differ because text color changes for dark mode
    assert b1 != b2


def test_cache_key_generation_and_fontsize_variation():
    # Same tex different fontsize should generate different image files
    ha._IMG_CACHE.clear()
    a = ha._render_tex_to_img("x^2", fontsize=12, dark_mode=False)
    ha._IMG_CACHE.clear()
    b = ha._render_tex_to_img("x^2", fontsize=18, dark_mode=False)

    assert a[0] != b[0]


def test_render_failure_uses_fallback(monkeypatch):
    # Simulate a failure inside the matplotlib figure/text path.

    def bad_figure(*args, **kwargs):
        class BadFigure(Figure):
            def savefig(self, *args, **kwargs):
                raise RuntimeError("Simulated save failure")

        return BadFigure()

    monkeypatch.setattr(ha, "Figure", bad_figure)

    path, render_w, render_h, display_w, display_h = ha._render_tex_to_img("\\invalid\\tex", fontsize=12, dark_mode=False)

    assert os.path.exists(path)
    assert (render_w, render_h) == (1, 1)
    assert display_w >= 0
    assert display_h >= 0
