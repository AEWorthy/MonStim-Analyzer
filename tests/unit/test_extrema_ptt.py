import numpy as np

from monstim_signals.transform.extrema import (
    calculate_exclusive_extrema_ptt_results,
    calculate_extrema_ptt_result,
    calculate_window_amplitude_results,
    detect_signal_extrema,
    make_window_span,
)


def test_extrema_ptt_includes_extrema_on_both_latency_flags():
    signal = np.array([0.0, 2.0, 0.0, -3.0, 0.0, 100.0])
    result = calculate_extrema_ptt_result(signal, make_window_span(0, "W", 1, 4, 1000), 1000)
    assert result.amplitude == 5.0
    assert (result.selected_max.sample_index, result.selected_min.sample_index) == (1, 3)


def test_independent_adjacent_windows_share_an_extremum_on_their_common_flag():
    signal = np.array([0.0, 2.0, 0.0, -3.0, 0.0, 1.0, 0.0])
    early = calculate_extrema_ptt_result(signal, make_window_span(0, "early", 0, 3, 1000), 1000)
    late = calculate_extrema_ptt_result(signal, make_window_span(1, "late", 3, 7, 1000), 1000)

    assert early.amplitude == 5.0
    assert (early.selected_max.sample_index, early.selected_min.sample_index) == (1, 3)
    assert late.amplitude == 4.0
    assert (late.selected_max.sample_index, late.selected_min.sample_index) == (5, 3)


def test_exclusive_adjacent_windows_resolve_a_shared_boundary_extremum_by_priority():
    signal = np.array([0.0, 2.0, 0.0, -3.0, 0.0, 1.0, 0.0])
    early, late = calculate_exclusive_extrema_ptt_results(
        signal,
        (make_window_span(0, "early", 0, 3, 1000), make_window_span(1, "late", 3, 7, 1000)),
        1000,
    )

    assert early.amplitude == 5.0
    assert late.amplitude == 0.0
    assert late.zero_reason == "single_extremum"
    assert late.excluded_owned_extrema_count == 1
    assert late.excluded_owner_window_indices == (0,)


def test_extrema_ptt_zero_and_invalid_semantics():
    span = make_window_span(0, "W", 0, 5, 1000)
    assert calculate_extrema_ptt_result(np.ones(5), span, 1000).zero_reason == "no_extrema"
    invalid = calculate_extrema_ptt_result(np.arange(5.0), make_window_span(0, "W", 0, 2, 1000), 1000)
    assert np.isnan(invalid.amplitude)
    assert invalid.zero_reason == "invalid_window"


def test_non_finite_gaps_are_not_paired():
    signal = np.array([0.0, 3.0, 0.0, np.nan, 0.0, -4.0, 0.0])
    result = calculate_extrema_ptt_result(signal, make_window_span(0, "W", 0, 7, 1000), 1000)
    assert result.amplitude == 0.0
    assert result.zero_reason == "no_opposite_pair"
    assert len(detect_signal_extrema(signal)) == 2


def test_independent_extrema_ptt_reuses_selected_extrema_in_overlapping_windows():
    signal = np.array([0.0, 0.0, 2.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0])
    spans = (
        make_window_span(0, "first", 0, 9, 1000, 0),
        make_window_span(1, "overlapping", 1, 8, 1000, 1),
    )

    first, overlapping = calculate_window_amplitude_results(signal, spans, 1000, "extrema_ptt")

    assert first.amplitude == overlapping.amplitude == 5.0
    assert (first.selected_max.sample_index, first.selected_min.sample_index) == (2, 4)
    assert (overlapping.selected_max.sample_index, overlapping.selected_min.sample_index) == (2, 4)
    assert first.excluded_owned_extrema_count == overlapping.excluded_owned_extrema_count == 0
    assert first.excluded_owner_window_indices == overlapping.excluded_owner_window_indices == ()


def test_exclusive_extrema_claims_only_selected_pair():
    signal = np.array([0.0, 2.0, 0.0, -3.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    spans = (make_window_span(0, "early", 0, 7, 1000, 0), make_window_span(1, "late", 2, 9, 1000, 1))
    early, late = calculate_exclusive_extrema_ptt_results(signal, spans, 1000)
    assert early.amplitude == 5.0
    assert late.amplitude == 2.0
    assert late.excluded_owned_extrema_count == 1
    assert late.excluded_owner_window_indices == (0,)
