# /monstim_signals/testing.py
"""
Script to test various functionalities of the MonStim Signals library, including CSV import and domain loading.
This script is intended for development and testing purposes, and should not be used in production.
"""
import logging


def test_csv_importer(overwrite_annot: bool = False):
    import os
    from pathlib import Path

    from monstim_signals.io.csv_importer import (csv_to_store, discover_by_ext,
                                                 infer_ds_ex, parse_session_rec)

    current_path = __file__
    base_path = Path.resolve(Path(current_path).parent.parent)
    data_path = os.path.join(base_path, "EXAMPLE DATA")
    store_root = Path(os.path.join(base_path, "data"))

    all_csv = discover_by_ext(Path(data_path), pattern="*.csv")

    experiments = {}

    for csv_path in all_csv:
        # Parse name/id information from the CSV path/filename
        sess, rec = parse_session_rec(csv_path)
        ds, ex = infer_ds_ex(csv_path, data_path)
        ds = ds or sess  # If no dataset name, use session name
        ex = ex or ds  # If no experiment name, use dataset name

        # Store the CSV path in a nested dictionary structure
        experiments.setdefault(ex, {}).setdefault(ds, {}).setdefault(sess, []).append(
            csv_path
        )

    # Print the discovered structure for debugging
    print(f"Found {len(all_csv)} CSV files in {data_path}:")
    for expt_name, ds_dict in experiments.items():
        for ds_name, ses_dict in ds_dict.items():
            for sess_name, csv_list in ses_dict.items():
                print(
                    f"  > Experiment: '{expt_name}', Dataset: '{ds_name}', Session: '{sess_name}', Recordings: {len(csv_list)}"
                )

    # Create the output directory structure
    experiments: dict[str, dict[str, dict[str, list[Path]]]] = experiments
    for expt, ds_dict in experiments.items():
        for ds, ses_dict in ds_dict.items():
            for sess, csv_list in ses_dict.items():
                for csv_path in csv_list:
                    out_dir = store_root / expt / ds / sess  # / csv_path.stem
                    out_dir.mkdir(parents=True, exist_ok=True)

                    filename = csv_path.stem
                    output_fp = out_dir / f"{filename}.ext"

                    # create the HDF5 file and meta JSON
                    csv_to_store(
                        csv_path,
                        output_fp,
                        overwrite_h5=True,
                        overwrite_meta=True,
                        overwrite_annot=overwrite_annot,
                    )


def test_domain_loading():
    """Load example Session, Dataset and Experiment and print summaries."""
    logging.info("Testing domain loading for Session, Dataset, and Experiment objects")
    from pathlib import Path

    from monstim_signals.io.repositories import (DatasetRepository,
                                                 ExperimentRepository,
                                                 SessionRepository)

    base = Path(__file__).resolve().parent.parent
    session_path = (
        base / "data" / "EMG-only data" / "240829 C328.1 post-dec mcurve_long-" / "RX35"
    )
    session = SessionRepository(session_path).load()
    print(
        f"Session ID: {session.id}, Recordings: {session.num_recordings}, Channels: {session.num_channels}, Scan Rate: {session.scan_rate} Hz"
    )
    print("Stim Amplitudes:", session.stimulus_voltages)
    print("Recordings:")
    for rec in session.recordings:
        print(f"  {rec.id}: {rec.num_channels} channels, {rec.scan_rate} Hz")

    dataset_path = (
        base / "data" / "EMG-only data" / "240829 C328.1 post-dec mcurve_long-"
    )
    dataset = DatasetRepository(dataset_path).load()
    print(dataset)
    for sess in dataset.sessions:
        print(sess.id, sess.num_recordings, sess.num_channels, sess.scan_rate)

    exp_path = base / "data" / "EMG-only data"
    exp = ExperimentRepository(exp_path).load()
    print(exp)
    for ds in exp.datasets:
        print(ds.id, ds.num_sessions)


def test_session_object():
    """Thoroughly test Session loading and plotting capabilities."""
    logging.info("Testing Session object loading and plotting routines")
    from pathlib import Path

    from monstim_signals.io.repositories import SessionRepository

    base = Path(__file__).resolve().parent.parent
    session_path = (
        base / "data" / "EMG-only data" / "240829 C328.1 post-dec mcurve_long-" / "RX35"
    )
    session = SessionRepository(session_path).load()

    # Print parameters and build caches
    session.session_parameters()
    _ = session.recordings_raw
    _ = session.recordings_filtered
    for idx in range(min(2, session.num_channels)):
        _ = session.get_m_max(method=session.default_method, channel_index=idx)

    # Run representative plotting functions
    logging.info("Plotting filtered EMG data for first two channels")
    session.plotter.plot_emg(channel_indices=[0, 1], data_type="filtered")
    logging.info("Plotting raw EMG data for first two channels")
    session.plotter.plot_emg(channel_indices=[0, 1], data_type="raw")
    logging.info("Plotting M-wave curves for first two channels")
    session.plotter.plot_mmax(channel_indices=[0, 1])
    logging.info("Plotting MaxH for first two channels")
    session.plotter.plot_reflexCurves(channel_indices=[0, 1])
    logging.info("Plotting M-wave curves with smoothing for first two channels")
    session.plotter.plot_m_curves_smoothened(channel_indices=[0, 1])


def test_dataset_object():
    """Load a Dataset, display parameters and test its plotting routines."""
    logging.info("Testing Dataset object loading and plotting routines")
    from pathlib import Path

    from monstim_signals.io.repositories import DatasetRepository

    base = Path(__file__).resolve().parent.parent
    dataset_path = (
        base / "data" / "EMG-only data" / "240829 C328.1 post-dec mcurve_long-"
    )
    dataset = DatasetRepository(dataset_path).load()

    dataset.dataset_parameters()
    logging.info("Plotting reflex curves for first two channels")
    dataset.plotter.plot_reflexCurves(channel_indices=[0, 1])
    logging.info("Plotting M-wave curves for first two channels")
    dataset.plotter.plot_mmax(channel_indices=[0, 1])
    logging.info("Plotting MaxH for first two channels")
    dataset.plotter.plot_maxH(channel_indices=[0, 1])


def test_experiment_object():
    """Load an Experiment, display parameters and test plotting routines."""
    logging.info("Testing Experiment object loading and plotting routines")
    from pathlib import Path

    from monstim_signals.io.repositories import ExperimentRepository

    base = Path(__file__).resolve().parent.parent
    exp_path = base / "data" / "EMG-only data"
    exp = ExperimentRepository(exp_path).load()

    exp.experiment_parameters()
    logging.info("Plotting reflex curves for first two channels")
    exp.plotter.plot_reflexCurves(channel_indices=[0, 1])
    logging.info("Plotting M-wave curves for first two channels")
    exp.plotter.plot_mmax(channel_indices=[0, 1])
    logging.info("Plotting MaxH for first two channels")
    exp.plotter.plot_maxH(channel_indices=[0, 1])


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # test_csv_importer(overwrite_annot=True) # works
    # test_domain_loading() # works
    test_session_object()  # works
    test_dataset_object()  # works
    test_experiment_object()  # works
