from .plot_options import (EMGOptions, SessionReflexCurvesOptions, SingleEMGRecordingOptions,
                           MMaxOptions, AverageReflexCurvesOptions, AverageSessionReflexOptions,
                           MaxHReflexOptions)

PLOT_NAME_DICT = {"EMG": "emg", "Suspected H-reflexes": "suspectedH", "Reflex Curves": "reflexCurves",
                  "M-max": "mmax", "Max H-reflex": "maxH", "Average Reflex Curves": "reflexCurves",
                  "Single EMG Recordings": "singleEMG", "Reflex Averages": "reflexAverages",
                  "Latency Window Trends": "latencyWindowTrends"}

PLOT_OPTIONS_DICT = {
            "session": {
                "EMG": EMGOptions,
                "Single EMG Recordings": SingleEMGRecordingOptions,
                "Reflex Averages": AverageSessionReflexOptions,
                "Reflex:Stimulus Curves": SessionReflexCurvesOptions,
                "Average Reflex:Stimulus Curve ": AverageSessionReflexOptions,
                "M-max": MMaxOptions
            },
            "dataset": {
                "Average Reflex:Stimulus Curves": AverageReflexCurvesOptions,
                "Max H-reflex": MaxHReflexOptions,
                "M-max": MMaxOptions
            },
            "experiment": {
                "Average Reflex:Stimulus Curves": AverageReflexCurvesOptions,
                "Max H-reflex": MaxHReflexOptions,
                "M-max": MMaxOptions
            }
        }