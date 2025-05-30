from dataclasses import dataclass

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
        if self.stim_duration is None:
            self.stim_duration = self.stim_delay + ((self.pulse_period + self.ramp_duration*2 + self.peak_duration) * self.num_pulses - (self.pulse_period + self.ramp_duration))
        if self.pulse_shape is None:
            self.pulse_shape = "Square"  # Default to Square if not specified
        if self.ramp_duration is None:
            self.ramp_duration = 0.0  # Default to 0 if not specified
        if self.stim_min_v is None:
            self.stim_min_v = 0.0  # Default to 0 if not specified
        if self.stim_max_v is not None and self.stim_max_v < self.stim_min_v:
            raise ValueError("Maximum stimulus voltage cannot be less than minimum voltage.")
        if self.stim_v < self.stim_min_v or self.stim_v > self.stim_max_v:
            raise ValueError("Stimulus voltage must be within the range of min and max voltages.")
        