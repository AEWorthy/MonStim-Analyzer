from monstim_gui.plotting import PLOT_NAME_DICT, PLOT_OPTIONS_DICT


def test_session_plot_options_do_not_expose_mmax():
    assert "M-max" not in PLOT_OPTIONS_DICT["session"]


def test_dataset_and_experiment_plot_options_still_expose_mmax():
    assert "M-max" in PLOT_OPTIONS_DICT["dataset"]
    assert "M-max" in PLOT_OPTIONS_DICT["experiment"]
    assert PLOT_NAME_DICT["M-max"] == "mmax"
