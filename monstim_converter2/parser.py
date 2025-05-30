import re
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any
from monstim_analysis.core.data_models import StimCluster

def detect_format(path: Path) -> str:
    """
    Inspect the first handful of lines to pick a format tag.
    Returns 'v3h' if it sees the new [Parameters]/[DATA] markers, else 'v3d'.
    """
    with path.open() as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            if line.strip().startswith('[Parameters]'):
                return 'v3h'
            if line.lower().startswith('file version'):
                return 'v3d'
    raise ValueError(f"Could not detect MonStim version for {path}. "
                     "Please ensure it is a valid MonStim CSV file.")

def parse(path: Path):
    version = detect_format(path)
    try:
        return PARSERS[version](path)
    except KeyError:
        raise RuntimeError(f"No parser for {version}")

# registry of parser functions
PARSERS = {}

CANONICAL_META = {
    'session_num': ['Session #', 'Session  0..99'], # used to be called session_id
    'num_emg_channels': ['# of Channels', 'No. Channels to acquire (N)'],
    'scan_rate': ['Scan Rate (Hz)', 'A/D Monitor Rate (Hz)'],
    'pre_stim_acquired': ['Pre-Stim Acq. time (ms)'],
    'post_stim_acquired': ['Post-Stim Acq. time (ms)'],
}

DEFAULT_META = {
    'amplitude_gain': 1.0,     # fallback if absent
    'force_gain':     100.0,   # example default
    # … any other fields you want guaranteed …
}

REVERSE_META_MAP = {
    syn.lower(): canon
    for canon, syns in CANONICAL_META.items()
    for syn in syns
}

def normalize_meta(raw_meta: dict[str,str]) -> dict[str,Any]:
    """
    Take the raw dict from parse_v1 or parse_v2 and return a new dict
    whose keys are the canonical names and whose values are converted
    to the appropriate Python types (int, float, str).
    """
    meta = {}
    for k, v in raw_meta.items():
        key = k.strip()
        canon = REVERSE_META_MAP.get(key.lower())
        if not canon:
            # if it’s completely unknown, carry it under its original name
            meta[key.lower().replace(' ', '_')] = v
            continue

        # convert strings to numbers when possible
        val: Any = v
        if v.isdigit():
            val = int(v)
        else:
            try:
                val = float(v)
            except ValueError:
                pass

        meta[canon] = val

    return meta

def register_parser(version):
    def decorator(fn):
        PARSERS[version] = fn
        return fn
    return decorator

@register_parser('v3d')
def parse_v3d(path: Path):
    # older format: no [DATA] tag
    raw_meta = {'monstim_version': 'v3d'}
    data_start = None
    with path.open() as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith('recorded'):
                data_start = i + 1
                break
            if ',' in line:
                k,v = line.split(',',1)
                raw_meta[k.strip()] = v.strip()
    df = pd.read_csv(path, sep=',', skiprows=data_start, header=None)
    data = df.values.astype('float32')
    meta = normalize_meta(raw_meta)

    # Set channel types and number of channels
    total_ch = data.shape[1]
    emg_ch = raw_meta.get('num_emg_channels', 0)
    types = ['emg'] * emg_ch + ['unknown'][:max(0, total_ch - emg_ch)]
    meta['channel_types'] = types
    meta['num_channels'] = int(total_ch)

    # Stimulus Cluster
    clusters = []
    clusters.append(StimCluster(
        stim_delay    = float(raw_meta['Start Delay (ms)']),
        stim_duration = None,
        stim_type     = 'Electrical',  # default type for v3d
        stim_v        = float(raw_meta['Stimulus Value (V)']),
        stim_min_v    = None,  # not provided in v3d format
        stim_max_v    = None,  # not provided in v3d format
        
        pulse_shape   = None,  # not provided in v3d format
        num_pulses    = int(raw_meta['# of Pulses in stim train']),
        pulse_period  = float(raw_meta['Stim Train pulse period (ms)']),
        peak_duration = float(raw_meta['Stimulus duration (ms)']),
        ramp_duration = None  # not provided in v3d format
    ))
    meta['stim_clusters'] = clusters
    
    # Set additional metadata
    meta['num_samples'] = int(data.shape[0])
    meta['emg_amp_gains'] = [int(float(raw_meta[f'EMG amp gain ch {i}'])) for i in range(1, int(float(meta['num_emg_channels'])) + 1)]


    return meta, data

@register_parser('v3h')
def parse_v3h(path: Path):
    # new format: metadata in [Parameters] block, then [DATA]
    raw_meta = {'monstim_version': 'v3h'}
    data_start = None
    with path.open() as f:
        for i, line in enumerate(f):
            if line.strip().startswith('[DATA]'):
                data_start = i + 1
                break
            if ',' in line:
                k,v = line.split(',',1)
                raw_meta[k.strip()] = v.strip()
    df = pd.read_csv(path, sep=',', skiprows=data_start, header=None)
    data = df.values.astype('float32')
    meta = normalize_meta(raw_meta)

    # Set channel types and number of channels
    total_ch = data.shape[1]
    emg_ch = meta.get('num_emg_channels', 0)
    types = ['emg'] * emg_ch + ['force', 'length'][:max(0, total_ch - emg_ch)]
    types += ['unknown'] * (total_ch - len(types))
    meta['channel_types'] = types
    meta['num_channels'] = int(total_ch)

    # Stimulus Clusters
    n = int(raw_meta['Stim specs cluster array.<size(s)>'])
    clusters = []
    for i in range(n):
        clusters.append(StimCluster(
            stim_delay    = float(raw_meta[f'Stim specs cluster array {i}.Start Delay (ms)']),
            stim_duration = None, # not provided in v3h format
            stim_type     = raw_meta[f'Stim type cluster array {i}.Stimulus Output'],
            stim_v        = float(raw_meta[f'Stim. {i}']),
            stim_min_v    = float(raw_meta[f'Stim specs cluster array {i}.Initial Stimulus ampl.']),
            stim_max_v    = float(raw_meta[f'Stim specs cluster array {i}.Maximum Stimulus']),
            
            pulse_shape   = raw_meta[f'Stim specs cluster array {i}.Pulse Shape'],
            num_pulses    = int(raw_meta[f'Stim specs cluster array {i}.#Pulses in train']),
            pulse_period  = float(raw_meta[f'Stim specs cluster array {i}.Total pulse period (ms)']),
            peak_duration = float(raw_meta[f'Stim specs cluster array {i}.Peak Level time (ms)']),
            ramp_duration       = float(raw_meta[f'Stim specs cluster array {i}.Ramp/Rel. time (ms)'])
        ))
    meta['stim_clusters'] = clusters

    
    # Set additional metadata
    meta['num_samples'] = int(data.shape[0])

    return meta, data
