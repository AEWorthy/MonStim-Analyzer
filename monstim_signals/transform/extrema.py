"""Deterministic extrema-based peak-to-trough amplitude calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from scipy.signal import find_peaks

ExtremumKind = Literal["max", "min"]


@dataclass(frozen=True)
class SignalExtremum:
    sample_index: int
    kind: ExtremumKind
    value: float
    finite_segment_id: int = 0


@dataclass(frozen=True)
class WindowSpan:
    window_index: int
    window_name: str
    priority_rank: int | None
    start_ms: float
    end_ms: float
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class WindowExtremaResult:
    window_index: int
    window_name: str
    priority_rank: int | None
    amplitude: float
    selected_max: SignalExtremum | None
    selected_min: SignalExtremum | None
    total_extrema_in_window: int
    available_extrema_in_window: int
    excluded_owned_extrema_count: int
    excluded_owner_window_indices: tuple[int, ...]
    zero_reason: str | None


@dataclass(frozen=True)
class ScalarWindowResult:
    window_index: int
    window_name: str
    priority_rank: int | None
    amplitude: float


def make_window_span(
    window_index: int, window_name: str, start_ms: float, end_ms: float, scan_rate: float, priority_rank: int | None = 0
) -> WindowSpan:
    return WindowSpan(
        window_index, window_name, priority_rank, float(start_ms), float(end_ms), int(start_ms * scan_rate / 1000), int(end_ms * scan_rate / 1000)
    )


def detect_signal_extrema(signal: np.ndarray) -> tuple[SignalExtremum, ...]:
    """Detect extrema separately in each contiguous finite part of *signal*."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    result: list[SignalExtremum] = []
    finite = np.isfinite(values)
    segment_id = 0
    start = 0
    while start < len(values):
        while start < len(values) and not finite[start]:
            start += 1
        end = start
        while end < len(values) and finite[end]:
            end += 1
        if end > start:
            part = values[start:end]
            maxima, _ = find_peaks(part)
            minima, _ = find_peaks(-part)
            result.extend(SignalExtremum(start + int(i), "max", float(values[start + i]), segment_id) for i in maxima)
            result.extend(SignalExtremum(start + int(i), "min", float(values[start + i]), segment_id) for i in minima)
            segment_id += 1
        start = end + 1
    return tuple(sorted(result, key=lambda item: item.sample_index))


def _invalid_result(span: WindowSpan, reason: str) -> WindowExtremaResult:
    return WindowExtremaResult(span.window_index, span.window_name, span.priority_rank, np.nan, None, None, 0, 0, 0, (), reason)


def _collapse_same_kind(extrema: Sequence[SignalExtremum]) -> list[SignalExtremum]:
    collapsed: list[SignalExtremum] = []
    for item in extrema:
        if not collapsed or item.kind != collapsed[-1].kind:
            collapsed.append(item)
        elif (item.kind == "max" and item.value > collapsed[-1].value) or (item.kind == "min" and item.value < collapsed[-1].value):
            collapsed[-1] = item
    return collapsed


def select_extrema_ptt_pair(
    extrema: Sequence[SignalExtremum], span: WindowSpan, *, claimed_by_sample: Mapping[int, int] | None = None
) -> WindowExtremaResult:
    """Select the deterministic best adjacent max/min pair for one window."""
    if span.end_sample - span.start_sample < 3:
        return _invalid_result(span, "invalid_window")
    # Extrema PTT treats the two visible latency flags as a closed interval.
    # Consequently, a boundary extremum is available to both adjacent or
    # overlapping windows in independent mode.  Exclusive mode subsequently
    # resolves that shared candidate through its ordered claim pass.
    inside = [item for item in extrema if span.start_sample <= item.sample_index <= span.end_sample]
    claimed_by_sample = claimed_by_sample or {}
    available = [item for item in inside if item.sample_index not in claimed_by_sample]
    owners = tuple(dict.fromkeys(claimed_by_sample[item.sample_index] for item in inside if item.sample_index in claimed_by_sample))
    if not inside:
        reason = "no_extrema"
    elif not available:
        reason = "all_candidate_extrema_claimed"
    elif len(available) == 1:
        reason = "single_extremum"
    else:
        reason = "no_opposite_pair"
    if len(available) < 2:
        return WindowExtremaResult(
            span.window_index,
            span.window_name,
            span.priority_rank,
            0.0,
            None,
            None,
            len(inside),
            len(available),
            len(inside) - len(available),
            owners,
            reason,
        )
    collapsed = _collapse_same_kind(sorted(available, key=lambda item: item.sample_index))
    candidates = [
        (left, right) for left, right in pairwise(collapsed) if left.kind != right.kind and left.finite_segment_id == right.finite_segment_id
    ]
    if not candidates:
        return WindowExtremaResult(
            span.window_index,
            span.window_name,
            span.priority_rank,
            0.0,
            None,
            None,
            len(inside),
            len(available),
            len(inside) - len(available),
            owners,
            "no_opposite_pair",
        )
    first, second = min(candidates, key=lambda pair: (-abs(pair[0].value - pair[1].value), pair[0].sample_index, pair[1].sample_index))
    maximum = first if first.kind == "max" else second
    minimum = first if first.kind == "min" else second
    return WindowExtremaResult(
        span.window_index,
        span.window_name,
        span.priority_rank,
        abs(maximum.value - minimum.value),
        maximum,
        minimum,
        len(inside),
        len(available),
        len(inside) - len(available),
        owners,
        None,
    )


def calculate_extrema_ptt_result(
    emg_data: np.ndarray, span: WindowSpan, scan_rate: float, *, precomputed_extrema: Sequence[SignalExtremum] | None = None
) -> WindowExtremaResult:
    data = np.asarray(emg_data, dtype=float).reshape(-1)
    if span.start_sample < 0 or span.end_sample > len(data):
        return _invalid_result(span, "window_out_of_bounds")
    if not np.isfinite(data[span.start_sample : min(span.end_sample + 1, len(data))]).any():
        return _invalid_result(span, "non_finite_data")
    return select_extrema_ptt_pair(precomputed_extrema if precomputed_extrema is not None else detect_signal_extrema(data), span)


def calculate_exclusive_extrema_ptt_results(
    emg_data: np.ndarray, spans: Sequence[WindowSpan], scan_rate: float, *, precomputed_extrema: Sequence[SignalExtremum] | None = None
) -> tuple[WindowExtremaResult, ...]:
    data = np.asarray(emg_data, dtype=float).reshape(-1)
    extrema = precomputed_extrema if precomputed_extrema is not None else detect_signal_extrema(data)
    claimed: dict[int, int] = {}
    results: dict[int, WindowExtremaResult] = {}
    valid = [span for span in spans if span.start_sample >= 0 and span.end_sample <= len(data) and span.end_sample - span.start_sample >= 3]
    for span in sorted(valid, key=lambda item: (item.start_sample, item.window_index)):
        baseline = calculate_extrema_ptt_result(data, span, scan_rate, precomputed_extrema=extrema)
        result = baseline if not claimed or np.isnan(baseline.amplitude) else select_extrema_ptt_pair(extrema, span, claimed_by_sample=claimed)
        results[span.window_index] = result
        if result.selected_max is not None and result.selected_min is not None:
            claimed[result.selected_max.sample_index] = span.window_index
            claimed[result.selected_min.sample_index] = span.window_index
    for span in spans:
        if span.window_index not in results:
            results[span.window_index] = _invalid_result(
                span, "window_out_of_bounds" if span.start_sample < 0 or span.end_sample > len(data) else "invalid_window"
            )
    return tuple(results[span.window_index] for span in spans)


def calculate_window_amplitude_results(
    emg_data: np.ndarray, spans: Sequence[WindowSpan], scan_rate: float, method: str
) -> tuple[WindowExtremaResult | ScalarWindowResult, ...]:
    if method == "exclusive_extrema_ptt":
        return calculate_exclusive_extrema_ptt_results(emg_data, spans, scan_rate)
    if method == "extrema_ptt":
        extrema = detect_signal_extrema(emg_data)
        return tuple(calculate_extrema_ptt_result(emg_data, span, scan_rate, precomputed_extrema=extrema) for span in spans)
    from .amplitude import calculate_emg_amplitude

    return tuple(
        ScalarWindowResult(
            span.window_index, span.window_name, span.priority_rank, calculate_emg_amplitude(emg_data, span.start_ms, span.end_ms, scan_rate, method)
        )
        for span in spans
    )
