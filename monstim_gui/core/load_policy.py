"""Immutable application loading and cache warm-up preferences."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WarmUpLevelPolicy:
    enabled: bool = False
    filtered_signals: bool = True
    methods: tuple[str, ...] = ()
    prepare_mmax: bool = False
    aggregates: bool = False


@dataclass(frozen=True)
class LoadPolicy:
    lazy_open_h5: bool = True
    parallel_loading: bool = True
    load_workers: int = 1
    session: WarmUpLevelPolicy = field(default_factory=WarmUpLevelPolicy)
    dataset: WarmUpLevelPolicy = field(default_factory=WarmUpLevelPolicy)
    experiment: WarmUpLevelPolicy = field(default_factory=WarmUpLevelPolicy)
