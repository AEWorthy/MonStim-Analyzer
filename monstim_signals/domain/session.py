# monstim_signals/domain/session.py
from typing import List, Any

from monstim_signals.domain.recording import Recording
from monstim_signals.core.data_models import LatencyWindow

class Session:
    """
    A collection of multiple Recordings, each at a different stimulus amplitude,
    all belonging to one “session” (animal & date).
    """
    def __init__(
        self,
        session_id : str,
        recordings : List[Recording],
        repo       : Any = None
    ):
        self.id = session_id
        self.recordings = recordings
        self.repo       = repo  # back‐pointer to SessionRepository

    @property
    def num_recordings(self) -> int:
        return len(self.recordings)
    @property
    def num_channels(self) -> int:
        # assume all recordings share num_channels; take from the first one
        return self.recordings[0].num_channels
    @property
    def scan_rate(self) -> int:
        # assume all recordings share scan_rate; take from the first one
        return self.recordings[0].scan_rate

    # ──────────────────────────────────────────────────────────────────
    # 1) Useful properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    def response_curve(self, channel: int, window: LatencyWindow) -> List[float]:
        # """
        # Example: for a given channel and a LatencyWindow (e.g. H-reflex),
        # compute the max value in that window minus baseline, for each recording.
        # Returns a list of [resp_at_stim1, resp_at_stim2, …] sorted by stim amplitude.
        # """
        # results = []
        # for rec in self.recordings:
        #     # 1) get the raw slice for that channel & window
        #     start = window.start_times[channel]
        #     end   = start + window.durations[channel]
        #     # Convert ms → sample index: (ms/1000)*scan_rate
        #     i0 = int(start/1000 * rec.scan_rate)
        #     i1 = int(end/1000 * rec.scan_rate)
        #     tr = rec.raw_view(ch=channel, t=slice(i0, i1))
        #     baseline = np.mean(rec.raw_view(ch=channel, t=slice(0, int(rec.scan_rate*window.start_times[channel]/1000))))
        #     resp = np.max(tr) - baseline
        #     results.append(resp)
        # return results
        raise NotImplementedError("response_curve method is not implemented yet.")

    @property
    def stim_amplitudes(self) -> List[float]:
        """
        Return a list of stimulus amplitudes for each recording in the session.
        This assumes that each recording's primary cluster stim_v is the amplitude for that recording.
        """
        return [rec.stim_amplitude for rec in self.recordings]

    # ──────────────────────────────────────────────────────────────────
    # 2) Exclude or include entire session (user action)
    # ──────────────────────────────────────────────────────────────────
    def exclude_session(self, do_exclude: bool = True):
        """
        If you want to exclude the entire session, propagate to every recording.
        """
        for rec in self.recordings:
            rec.exclude_recording(do_exclude)
        # No separate session‐level annotation file for now; each rec holds its own.
        # GUI should handle this by checking each rec.annot.excluded 
        # and not showing sessions if all recs are excluded.
        if self.repo is not None:
            self.repo.save(self)   # calls each rec.repo.save(rec)

    # ──────────────────────────────────────────────────────────────────
    # 3) Clean‐up
    # ──────────────────────────────────────────────────────────────────
    def close(self):
        for rec in self.recordings:
            rec.close()
    # ──────────────────────────────────────────────────────────────────
    # 4) Object representation
    # ──────────────────────────────────────────────────────────────────
    def __repr__(self):
        return f"Session(session_id={self.id}, num_recordings={self.num_recordings})"
    def __str__(self):
        return f"Session: {self.id} with {self.num_recordings} recordings"
    def __len__(self):
        return self.num_recordings
    
