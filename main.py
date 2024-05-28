import tkinter as tk
from tkinter import filedialog, ttk
import os
from monstim_to_pickle import pickle_data
import utils
import Analyze_EMG

class EMGAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG Analysis GUI")
        self.geometry("800x600")
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Set default input and output paths
        self.default_input_path = os.path.join(script_dir, 'files_to_analyze')
        self.default_output_path = os.path.join(script_dir, 'output')

        self.create_widgets()

    def create_widgets(self):
        # Create frames
        input_frame = tk.Frame(self)
        input_frame.pack(pady=10)

        output_frame = tk.Frame(self)
        output_frame.pack(pady=10)

        analysis_frame = tk.Frame(self)
        analysis_frame.pack(pady=10)

        # Input frame widgets
        input_label = tk.Label(input_frame, text="Input Directory:")
        input_label.pack(side=tk.LEFT)

        self.input_entry = tk.Entry(input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT)

        input_button = tk.Button(input_frame, text="Browse", command=self.browse_input_dir)
        input_button.pack(side=tk.LEFT)

        # Output frame widgets
        output_label = tk.Label(output_frame, text="Output Directory:")
        output_label.pack(side=tk.LEFT)

        self.output_entry = tk.Entry(output_frame, width=40)
        self.output_entry.pack(side=tk.LEFT)

        output_button = tk.Button(output_frame, text="Browse", command=self.browse_output_dir)
        output_button.pack(side=tk.LEFT)

        # Set default values for input and output entries
        self.input_entry.insert(0, self.default_input_path)
        self.output_entry.insert(0, self.default_output_path)

        # Analysis frame widgets
        process_button = tk.Button(analysis_frame, text="Process Data", command=self.process_data)
        process_button.pack(side=tk.LEFT)

        self.dataset_combo = ttk.Combobox(analysis_frame, state="readonly")
        self.dataset_combo.pack(side=tk.LEFT)

        self.session_combo = ttk.Combobox(analysis_frame, state="readonly")
        self.session_combo.pack(side=tk.LEFT)

        analyze_button = tk.Button(analysis_frame, text="Analyze", command=self.analyze_data)
        analyze_button.pack(side=tk.LEFT)

        # Add a checkbox for overwriting pickled files
        self.overwrite_var = tk.BooleanVar()
        overwrite_checkbox = tk.Checkbutton(analysis_frame, text="Overwrite Pickled Files", variable=self.overwrite_var)
        overwrite_checkbox.pack(side=tk.LEFT)

    def browse_input_dir(self):
        input_dir = filedialog.askdirectory()
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, input_dir)

    def browse_output_dir(self):
        output_dir = filedialog.askdirectory()
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, output_dir)

    def process_data(self):
        input_dir = self.input_entry.get()
        output_dir = self.output_entry.get()

        if input_dir and output_dir:

            # Check if the output directory and its subdirectories contain pickled files
            pickled_files = []
            for root, _, files in os.walk(output_dir):
                pickled_files.extend([os.path.join(root, f) for f in files if f.endswith('.pickle')])

            # If pickled files exist and overwrite is not selected, load them directly
            if pickled_files and not self.overwrite_var.get():
                self.dataset_dict, self.datasets = utils.unpackPickleOutput(output_dir)
                self.update_dataset_combo()
            else:
                # If pickled files don't exist or overwrite is selected, process the CSV files
                pickle_data(input_dir, output_dir)
                self.dataset_dict, self.datasets = utils.unpackPickleOutput(output_dir)
                self.update_dataset_combo()

    def update_dataset_combo(self):
        self.dataset_combo["values"] = [f"Dataset {idx}: {dataset}" for idx, dataset in enumerate(self.datasets)]

    def update_session_combo(self, dataset_idx):
        dataset = self.dataset_dict[self.datasets[dataset_idx]]
        sessions = [session.name for session in dataset.sessions]
        self.session_combo["values"] = sessions

    def analyze_data(self):
        dataset_idx = self.dataset_combo.current()
        session_idx = self.session_combo.current()

        if dataset_idx >= 0 and session_idx >= 0:
            dataset_oi = utils.dataset_oi(self.dataset_dict, self.datasets, dataset_idx)
            session_oi = utils.session_oi(self.dataset_dict, self.datasets, dataset_idx, session_idx)

            # Add your analysis code here
            # For example:
            channel_names = ["LG", "TA"]
            session_oi.plot_emg(channel_names=channel_names, m_flags=True, h_flags=True, data_type='filtered')
            session_oi.plot_reflex_curves(channel_names=channel_names, method='rms')
            dataset_oi.plot_reflex_curves(channel_names=channel_names, method='rms')

if __name__ == "__main__":
    app = EMGAnalysisGUI()
    app.mainloop()