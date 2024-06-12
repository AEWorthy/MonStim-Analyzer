import tkinter as tk
from tkinter import filedialog, ttk
import os
import io
from Analyze_EMG import EMGSession, EMGDataset, unpackPickleOutput, getDatasetInfo, dataset_oi

class EMGAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG Analysis GUI")
        self.geometry("800x600")

        # Set the output folder path
        self.output_folder = os.path.join(os.getcwd(), "output")

        # Unpack the pickled outputs
        self.dataset_dict, self.dataset_names = unpackPickleOutput(self.output_folder)

        # Create tabs
        self.tabControl = ttk.Notebook(self)
        self.session_tab = ttk.Frame(self.tabControl)
        self.dataset_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.session_tab, text="Session Analysis")
        self.tabControl.add(self.dataset_tab, text="Dataset Analysis")
        self.tabControl.pack(expand=1, fill="both")

        # Create widgets for Session Analysis tab
        self.session_label = ttk.Label(self.session_tab, text="Select a session:")
        self.session_label.pack(pady=10)
        self.session_combo = ttk.Combobox(self.session_tab, values=self.get_all_sessions(), state="readonly")
        self.session_combo.pack()
        self.session_analyze_button = ttk.Button(self.session_tab, text="Analyze Session", command=self.analyze_session)
        self.session_analyze_button.pack(pady=10)
        self.session_output = tk.Text(self.session_tab, height=10, width=60)
        self.session_output.pack(pady=10)

        # Create widgets for plotting session data
        self.session_plot_frame = ttk.Frame(self.session_tab)
        self.session_plot_frame.pack(pady=10)
        self.session_plot_type_label = ttk.Label(self.session_plot_frame, text="Plot Type:")
        self.session_plot_type_label.grid(row=0, column=0, padx=5)
        self.session_plot_type_combo = ttk.Combobox(self.session_plot_frame, values=["emg", "suspectedH", "mmax", "reflexCurves"], state="readonly")
        self.session_plot_type_combo.grid(row=0, column=1, padx=5)
        self.session_plot_button = ttk.Button(self.session_plot_frame, text="Plot Data", command=self.plot_session_data)
        self.session_plot_button.grid(row=0, column=2, padx=5)

        # Create widgets for Dataset Analysis tab
        self.dataset_label = ttk.Label(self.dataset_tab, text="Select a dataset:")
        self.dataset_label.pack(pady=10)
        self.dataset_combo = ttk.Combobox(self.dataset_tab, values=self.dataset_names, state="readonly")
        self.dataset_combo.pack()
        self.dataset_analyze_button = ttk.Button(self.dataset_tab, text="Analyze Dataset", command=self.analyze_dataset)
        self.dataset_analyze_button.pack(pady=10)
        self.dataset_output = tk.Text(self.dataset_tab, height=10, width=60)
        self.dataset_output.pack(pady=10)

        # Create widgets for plotting dataset data
        self.dataset_plot_frame = ttk.Frame(self.dataset_tab)
        self.dataset_plot_frame.pack(pady=10)
        self.dataset_plot_type_label = ttk.Label(self.dataset_plot_frame, text="Plot Type:")
        self.dataset_plot_type_label.grid(row=0, column=0, padx=5)
        self.dataset_plot_type_combo = ttk.Combobox(self.dataset_plot_frame, values=["reflexCurves", "mmax", "maxH"], state="readonly")
        self.dataset_plot_type_combo.grid(row=0, column=1, padx=5)
        self.dataset_plot_button = ttk.Button(self.dataset_plot_frame, text="Plot Data", command=self.plot_dataset_data)
        self.dataset_plot_button.grid(row=0, column=2, padx=5)

    def get_all_sessions(self):
        all_sessions = []
        for dataset in self.dataset_dict.values():
            for session in dataset:
                all_sessions.append(session)
        return all_sessions

    def analyze_session(self):
        session_file = self.session_combo.get()
        if session_file:
            session = EMGSession(session_file)
            self.session_output.delete('1.0', tk.END)
            self.session_output.insert(tk.END, f"Analyzing session: {session.session_name}\n\n")
            # Capture the output of session_parameters
            captured_output = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output
            session.session_parameters()
            sys.stdout = original_stdout
            output_str = captured_output.getvalue()
            self.session_output.insert(tk.END, output_str)

    def plot_session_data(self):
        session_file = self.session_combo.get()
        if session_file:
            session = EMGSession(session_file)
            plot_type = str(self.session_plot_type_combo.get())
            session.plot(plot_type=plot_type)

    def analyze_dataset(self):
        selected_dataset = self.dataset_combo.get()
        if selected_dataset:
            date, animal_id, condition = getDatasetInfo(selected_dataset)
            dataset = dataset_oi(self.dataset_dict, self.dataset_names, self.dataset_names.index(selected_dataset))
            self.dataset_output.delete('1.0', tk.END)
            self.dataset_output.insert(tk.END, f"Analyzing dataset: {selected_dataset}\n\n")
            # Capture the output of dataset_parameters
            captured_output = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output
            dataset.dataset_parameters()
            sys.stdout = original_stdout
            output_str = captured_output.getvalue()
            self.dataset_output.insert(tk.END, output_str)

    def plot_dataset_data(self):
        selected_dataset = self.dataset_combo.get()
        if selected_dataset:
            date, animal_id, condition = getDatasetInfo(selected_dataset)
            dataset = dataset_oi(self.dataset_dict, self.dataset_names, self.dataset_names.index(selected_dataset))
            plot_type = str(self.dataset_plot_type_combo.get())
            dataset.plot(plot_type=plot_type)

if __name__ == "__main__":
    import sys
    channel_names = []
    app = EMGAnalysisGUI()
    app.mainloop()