# /monstim_signals/testing.py
'''
Script to test various functionalities of the MonStim Signals library, including CSV import and domain loading.
This script is intended for development and testing purposes, and should not be used in production.
'''

def test_csv_importer():
    import os
    from monstim_signals.io.csv_importer import discover_by_ext, parse_session_rec, infer_ds_ex, csv_to_store
    from pathlib import Path
    
    current_path = __file__
    base_path = Path.resolve(Path(current_path).parent.parent)
    data_path = os.path.join(base_path, 'EXAMPLE DATA')
    store_root = Path(os.path.join(base_path, 'data_store'))

    all_csv = discover_by_ext(Path(data_path), pattern='*.csv')

    experiments = {}

    for csv_path in all_csv:
        # Parse name/id information from the CSV path/filename
        sess, rec = parse_session_rec(csv_path)
        ds, ex = infer_ds_ex(csv_path, data_path)
        ds = ds or sess # If no dataset name, use session name
        ex = ex or ds # If no experiment name, use dataset name
        
        # Store the CSV path in a nested dictionary structure
        experiments.setdefault(ex, {})\
               .setdefault(ds, {})\
               .setdefault(sess, [])\
               .append(csv_path)
    
    # Print the discovered structure for debugging
    print(f"Found {len(all_csv)} CSV files in {data_path}:")
    for expt_name, ds_dict in experiments.items():
        for ds_name, ses_dict in ds_dict.items():
            for sess_name, csv_list in ses_dict.items():
                print(f"  > Experiment: '{expt_name}', Dataset: '{ds_name}', Session: '{sess_name}', Recordings: {len(csv_list)}")

    # Create the output directory structure
    experiments: dict[str, dict[str, dict[str, list[Path]]]] = experiments
    for expt, ds_dict in experiments.items():
        for ds, ses_dict in ds_dict.items():
            for sess, csv_list in ses_dict.items():
                for csv_path in csv_list:
                    out_dir = store_root / expt / ds / sess# / csv_path.stem
                    out_dir.mkdir(parents=True, exist_ok=True)

                    filename = csv_path.stem
                    output_fp    = out_dir / f"{filename}.ext"

                    # create the HDF5 file and meta JSON
                    csv_to_store(csv_path, output_fp, overwrite_h5=True, overwrite_meta=True, overwrite_annot=False)

def test_domain_loading():
    from pathlib import Path
    from monstim_signals.io.repositories import SessionRepository, DatasetRepository, ExperimentRepository

    session_path = Path(r'C:\Users\aewor\Documents\GitHub\MonStim_Analysis\data_store\EMG-only data\240829 C328.1 post-dec mcurve_long-\RX35')
    session = SessionRepository(session_path).load()
    print(f"Session ID: {session.id}, Recordings: {session.num_recordings}, Channels: {session.num_channels}, Scan Rate: {session.scan_rate} Hz")
    print("Stim Amplitudes:", session.stim_amplitudes)
    print("Recordings:")
    for rec in session.recordings:
        print(f"  {rec.id}: {rec.num_channels} channels, {rec.scan_rate} Hz")

    dataset = DatasetRepository(Path(r'C:\Users\aewor\Documents\GitHub\MonStim_Analysis\data_store\EMG-only data\240829 C328.1 post-dec mcurve_long-')).load()
    print(dataset)
    for sess in dataset.sessions:
        print(sess.id, sess.num_recordings, sess.num_channels, sess.scan_rate)


    exp = ExperimentRepository(Path(r'C:\Users\aewor\Documents\GitHub\MonStim_Analysis\data_store\EMG-only data')).load()
    print(exp)
    for ds in exp.datasets:
        print(ds.id, ds.num_sessions)

def test_session_object():
    from monstim_signals.io.repositories import SessionRepository

    session_path = Path(r'C:\Users\aewor\Documents\GitHub\MonStim_Analysis\data_store\EMG-only data\240829 C328.1 post-dec mcurve_long-\RX35')
    session = SessionRepository(session_path).load()
    
    session.session_parameters()
    session.m_max_report()
    session.plot(plot_type="mmax")
    session.plot(plot_type="reflexCurves", channel=1, all_flags=True)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # test_csv_importer()
    # test_domain_loading()
    test_session_object()
