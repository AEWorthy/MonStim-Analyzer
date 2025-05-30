from dataclasses import dataclass
from typing import List
from matplotlib.lines import Line2D
from monstim_signals.core.utils import DATA_VERSION



# To do: Add a method to create dataset latency window objects for each session in the dataset. Make the default windows be the m-wave and h-reflex windows.
@dataclass
class LatencyWindow:
    name: str
    color: str
    start_times: List[float]
    durations: List[float]
    linestyle: str = '--'
    window_version: str = DATA_VERSION

    @property
    def end_times(self):
        return [start + dur for start, dur in zip(self.start_times, self.durations)]

    def plot(self, ax, channel_index):
        start_exists = end_exists = False
        
        for line in ax.lines:
            if isinstance(line, Line2D):
                if line.get_xdata()[0] == self.start_times[channel_index] and line.get_color() == self.color:
                    start_exists = True
                elif line.get_xdata()[0] == self.end_times[channel_index] and line.get_color() == self.color:
                    end_exists = True
                
                if start_exists and end_exists:
                    break
        
        if not start_exists:
            ax.axvline(self.start_times[channel_index], color=self.color, linestyle=self.linestyle)
        
        if not end_exists:
            ax.axvline(self.end_times[channel_index], color=self.color, linestyle=self.linestyle)

    def get_legend_element(self, stylized=True):
        if stylized:
            return Line2D([0], [0], color=self.color, linestyle=self.linestyle, label=self.name)
        else:
            return Line2D([0], [0], color=self.color, linestyle='-', label=self.name)

@dataclass
class StimCluster:
    stim_delay:    float  # train start delay in ms
    stim_duration: float  # total train duration in ms.
    stim_type:     str    # e.g. "Electrical", "OFF", "Motor Length"
    stim_v:        float  # initial amplitude in V
    stim_min_v:    float  # minimum amplitude that the cluster can reach in V
    stim_max_v:    float  # maximum amplitude that the cluster can reach in V
    
    pulse_shape:   str    # e.g. "Square", "Triangle", "Sine"
    num_pulses:    int    # number of pulses in the train
    pulse_period:  float  # total pulse period in ms
    peak_duration: float # ms, time from start of pulse to peak amplitude
    ramp_duration: float  # ms, time to ramp up to peak amplitude
    def __post_init__(self):
        if self.ramp_duration is None:
            self.ramp_duration = 0.0
        if self.stim_duration is None:
            self.stim_duration = self.stim_delay + ((self.pulse_period + self.ramp_duration*2 + self.peak_duration) * self.num_pulses - (self.pulse_period + self.ramp_duration))
        if self.pulse_shape is None:
            self.pulse_shape = "Square"  # Default to Square if not specified
        if self.ramp_duration is None:
            self.ramp_duration = 0.0  # Default to 0 if not specified
        