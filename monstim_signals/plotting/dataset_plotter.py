import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, List
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from monstim_signals.plotting.base_plotter import BasePlotter, UnableToPlotError

if TYPE_CHECKING:
    from monstim_signals.domain.dataset import Dataset

class DatasetPlotter(BasePlotter):
    """
    A class for EMG data from an EMGDataset object.

    Args:
        dataset (EMGDataset): The EMG dataset object to be imported.

    Attributes:
        dataset (EMGDataset): The imported EMG dataset object.

    Methods:
        plot_reflex_curves: Plots the average M-response and H-reflex curves for each channel.
        plot_max_h_reflex: Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.
        help: Displays the help text for the class.
    """
    def __init__(self, dataset : 'Dataset'):
        from monstim_signals.domain.dataset import Dataset
        if not isinstance(dataset, Dataset):
            raise TypeError("DatasetPlotter requires an instance of Dataset.")
        self.emg_object : 'Dataset' = dataset
        super().__init__(self.emg_object)
        self.set_plot_defaults()
    
    def help(self):
        help_text = """
        EMGDatasetPlotter class for plotting EMG data from an EMGDataset object.
        ========================================================================
        Methods:
        1. plot_reflex_curves: Plots the average M-response and H-reflex curves for each channel.
            -Example:
                plot_reflex_curves(channel_names=['Channel 1', 'Channel 2'], method='rms', relative_to_mmax=False, manual_mmax=None)
        
        2. plot_max_h_reflex: Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.
            -Example:
                plot_max_h_reflex(channel_names=['Channel 1', 'Channel 2'], method='rms', relative_to_mmax=False, manual_mmax=None)
        
        3. help: Displays this help text.
        ========================================================================

        """
        print(help_text)

    # Dataset plotting functions
    def plot_reflexCurves(self, channel_indices : List[int] = None, method=None, plot_legend=True, relative_to_mmax=False, manual_mmax=None, canvas=None):
        """
        Plots the M-response and H-reflex curves for each channel.

        Args:
            channel_names (list): A list of custom channel names. If specified, the channel names will be used in the plot titles.
            method (str): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.

        Returns:
            None
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method
        channel_names = self.emg_object.channel_names
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas)

        # Get unique binned stimulus voltages
        stimulus_voltages = self.emg_object.stimulus_voltages

        raw_data_dict = {
            'channel_index': [],
            'stimulus_v': [],
            'avg_m_wave': [],
            'stderr_m_wave': [],
            'avg_h_wave': [],
            'stderr_h_wave': []
        }

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            m_wave_means, m_wave_error = self.emg_object.get_avg_m_wave_amplitudes(method, channel_index)
            h_response_means, h_response_error = self.emg_object.get_avg_h_wave_amplitudes(method, channel_index)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    channel_m_max = manual_mmax[channel_index]
                else:
                    channel_m_max = self.emg_object.get_avg_m_max(method, channel_index)
                    if channel_m_max is None:
                        raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                try:
                    m_wave_means = [(amplitude / channel_m_max) for amplitude in m_wave_means]
                    m_wave_error = [(amplitude / channel_m_max) for amplitude in m_wave_error]
                    h_response_means = [(amplitude / channel_m_max) for amplitude in h_response_means]
                    h_response_error = [(amplitude / channel_m_max) for amplitude in h_response_error]
                except ZeroDivisionError:
                    raise UnableToPlotError(f'M-max for channel {channel_index} is zero. Cannot divide by zero.')

            # Append data to raw data dictionary - will be relative to M-max if specified.
            raw_data_dict['channel_index'].extend([channel_index]*len(stimulus_voltages))
            raw_data_dict['stimulus_v'].extend(stimulus_voltages)
            raw_data_dict['avg_m_wave'].extend(m_wave_means)
            raw_data_dict['stderr_m_wave'].extend(m_wave_error)
            raw_data_dict['avg_h_wave'].extend(h_response_means)
            raw_data_dict['stderr_h_wave'].extend(h_response_error)
            
            # Plot the M-wave and H-response amplitudes for each channel
            if num_channels == 1:
                ax.plot(stimulus_voltages, m_wave_means, color=self.emg_object.m_color, label='M-wave')
                ax.fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_error), np.array(m_wave_means) + np.array(m_wave_error), color='r', alpha=0.2)
                ax.plot(stimulus_voltages, h_response_means, color=self.emg_object.h_color, label='H-response')
                ax.fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_error), np.array(h_response_means) + np.array(h_response_error), color='b', alpha=0.2)
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                if plot_legend:
                    ax.legend()
            else:
                axes[channel_index].plot(stimulus_voltages, m_wave_means, color=self.emg_object.m_color, label='M-wave')
                axes[channel_index].fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_error), np.array(m_wave_means) + np.array(m_wave_error), color='r', alpha=0.2)
                axes[channel_index].plot(stimulus_voltages, h_response_means, color=self.emg_object.h_color, label='H-response')
                axes[channel_index].fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_error), np.array(h_response_means) + np.array(h_response_error), color='b', alpha=0.2)
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                if plot_legend:
                    axes[channel_index].legend()

        # Set labels and title
        fig.suptitle('M-response and H-reflex Curves')
        if num_channels == 1:
            ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'Reflex Ampl. (M-max, {method})')
        else:
            fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'Reflex Ampl. (M-max, {method})')

        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'stimulus_v'], inplace=True)
        return raw_data_df

    def plot_maxH(self, channel_indices : List[int] = None, method=None, relative_to_mmax=False, manual_mmax=None, max_stim_value=None, bin_margin=0, canvas=None):
        """
        Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.

        Args:
            channel_names (list): List of custom channel names. Default is an empty list.
            method (str): Method for calculating the amplitude. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
            relative_to_mmax (bool): Flag indicating whether to make the M-wave amplitudes relative to the maximum M-wave amplitude. Default is False.
            manual_mmax (float): Manual value for the maximum M-wave amplitude. Default is None.

        Returns:
            None
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method
        channel_names = self.emg_object.channel_names
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas, figsizes='small')

        raw_data_dict = {
            'channel_index': [],
            'stimulus_v': [],
            'm_wave_amplitudes': [],
            'h_wave_amplitudes': [],
        }

        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            # Get average H-wave amplitudes
            h_response_means, _ = self.emg_object.get_avg_h_wave_amplitudes(method, channel_index)
            stimulus_voltages = self.emg_object.stimulus_voltages

            # Filter out stimulus voltages greater than max_stim_value if specified
            if max_stim_value is not None:
                filtered_indices = [i for i, v in enumerate(stimulus_voltages) if v <= max_stim_value]
                stimulus_voltages = [stimulus_voltages[i] for i in filtered_indices]
                h_response_means = [h_response_means[i] for i in filtered_indices]

            # Find the voltage with the maximum average H-reflex amplitude
            max_h_reflex_amplitude = max(h_response_means)
            max_h_reflex_voltage = stimulus_voltages[h_response_means.index(max_h_reflex_amplitude)]
            
            # Define the range of voltages around the max H-reflex voltage
            voltage_indices = range(max(0, h_response_means.index(max_h_reflex_amplitude) - bin_margin),
                                min(len(stimulus_voltages), h_response_means.index(max_h_reflex_amplitude) + bin_margin + 1))
            marginal_voltages = [stimulus_voltages[i] for i in voltage_indices]

            # Collect M-wave and H-response amplitudes for the marginal bins
            stimulus_voltages = []
            m_wave_amplitudes = []
            h_response_amplitudes = []
            for voltage in marginal_voltages:
                m_waves = self.emg_object.get_m_wave_amplitudes_at_voltage(method, channel_index, voltage)
                h_responses = self.emg_object.get_h_wave_amplitudes_at_voltage(method, channel_index, voltage)
                m_wave_amplitudes.extend(m_waves)
                h_response_amplitudes.extend(h_responses)
                stimulus_voltages.extend([voltage] * len(m_waves))

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax[channel_index]
                else:
                    m_max = self.emg_object.get_avg_m_max(method=method, channel_index=channel_index)
                try:
                    m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                    h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]
                except TypeError:
                    raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                except ZeroDivisionError:
                    raise UnableToPlotError(f'M-max is zero for channel {channel_index}. Cannot divide by zero.')

            # # Crop data if max_stim_value is specified
            # if max_stim_value is not None:
            #     mask = np.array(stimulus_voltages) <= max_stim_value
            #     stimulus_voltages = np.array(stimulus_voltages)[mask]
            #     m_wave_amplitudes = np.array(m_wave_amplitudes)[mask]
            #     h_response_amplitudes = np.array(h_response_amplitudes)[mask]

            # Append data to raw data dictionary
            raw_data_dict['channel_index'].extend([channel_index] * len(m_wave_amplitudes))
            raw_data_dict['stimulus_v'].extend(stimulus_voltages)
            raw_data_dict['m_wave_amplitudes'].extend(m_wave_amplitudes)
            raw_data_dict['h_wave_amplitudes'].extend(h_response_amplitudes)

            # Plot the M-wave and H-response amplitudes for the maximum H-reflex voltage.
            m_x = 1
            h_x = 2.5
            if num_channels == 1:
                ax.plot(m_x, [m_wave_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_wave_amplitudes)}\navg. = {np.average(m_wave_amplitudes):.2f}\nstd. = {np.std(m_wave_amplitudes):.2f}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes)), ha='left', color=self.emg_object.m_color)
                ax.errorbar(m_x, np.mean(m_wave_amplitudes), yerr=np.std(m_wave_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.plot(h_x, [h_response_amplitudes], color=self.emg_object.h_color, marker='o', markersize=5)
                ax.annotate(f'n={len(h_response_amplitudes)}\navg. = {np.average(h_response_amplitudes):.2f}\nstd. = {np.std(h_response_amplitudes):.2f}', xy=(h_x + 0.4, np.mean(h_response_amplitudes)), ha='left', color=self.emg_object.h_color)
                ax.errorbar(h_x, np.mean(h_response_amplitudes), yerr=np.std(h_response_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticks([m_x, h_x])
                ax.set_xticklabels(['M-response', 'H-reflex'])
                ax.set_title(f'{channel_names[0]} ({round(max_h_reflex_voltage, 2)} ± {round((self.emg_object.bin_size/2)+(self.emg_object.bin_size * bin_margin),2)}V)')
                ax.set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
            else:
                axes[channel_index].plot(m_x, [m_wave_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_wave_amplitudes)}\navg. = {np.average(m_wave_amplitudes):.2f}\nstd. = {np.std(m_wave_amplitudes):.2f}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes)), ha='left', color=self.emg_object.m_color)
                axes[channel_index].errorbar(m_x, np.mean(m_wave_amplitudes), yerr=np.std(m_wave_amplitudes), color='black', marker='+', markersize=10, capsize=10)

                axes[channel_index].plot(h_x, [h_response_amplitudes], color=self.emg_object.h_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(h_response_amplitudes)}\navg. = {np.average(h_response_amplitudes):.2f}\nstd. = {np.std(h_response_amplitudes):.2f}', xy=(h_x + 0.4, np.mean(h_response_amplitudes)), ha='left', color=self.emg_object.h_color)
                axes[channel_index].errorbar(h_x, np.mean(h_response_amplitudes), yerr=np.std(h_response_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_title(f'{channel_names[channel_index]} ({round(max_h_reflex_voltage, 2)} ± {round((self.emg_object.bin_size/2)+(self.emg_object.bin_size * bin_margin),2)}V)')
                axes[channel_index].set_xticks([m_x, h_x])
                axes[channel_index].set_xticklabels(['M-response', 'H-reflex'])
                axes[channel_index].set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.

        # Set labels and title
        fig.suptitle('EMG Responses at Max H-reflex Stimulation')
        if num_channels == 1:
            ax.set_xlabel('Response Type')
            ax.set_ylabel(f'EMG Amp. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'EMG Amp. (M-max, {method})')
        else:
            fig.supxlabel('Response Type')
            fig.supylabel(f'EMG Amp. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'EMG Amp. (M-max, {method})')

        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'stimulus_v'], inplace=True)
        return raw_data_df

    def plot_mmax(self, channel_indices : List[int] = None, method : str = None, canvas : FigureCanvas = None):
        """
        Plots the average M-max of each session for each channel.

        Args:
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.

        Returns:
            None
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method

        channel_names = self.emg_object.channel_names

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas)

        all_m_max_amplitudes = []
        raw_data_dict = {
            'channel_index': [],
            'session_id': [],
            'm_max_threshold': [],
            'm_max_amplitude': [],
        }

        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            m_max_amplitudes = []
            m_max_thresholds = []
            session_ids = []
            for session in self.emg_object.emg_sessions:
                try:
                    m_max, mmax_low_stim, _ = session.get_m_max(method=method, channel_index=channel_index, return_mmax_stim_range=True)
                    m_max_amplitudes.append(m_max)
                    m_max_thresholds.append(mmax_low_stim)
                    session_ids.append(session.session_id)
                except IndexError:
                    m_max_amplitudes.append(np.nan)
                    m_max_thresholds.append(np.nan)
                    session_ids.append(session.session_id)
                
            # Append M-wave amplitudes to superlist for y-axis adjustment.
            all_m_max_amplitudes.extend(m_max_amplitudes)            

            # Plot the M-wave amplitudes for each session
            m_x = 1
            if num_channels == 1:
                ax.plot(m_x, [m_max_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_max_amplitudes)}\nAvg. M-max: {np.mean(m_max_amplitudes):.2f}mV\nStdev. M-Max: {np.std(m_max_amplitudes):.2f}mV\nAvg. Stim.: above {np.mean(m_max_thresholds):.2f} mV', xy=(m_x + 0.2, np.mean(m_max_amplitudes)), ha='left', va='center', color='black')
                ax.errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticklabels(['M-response'])
                ax.set_title(f'{channel_names[0]}')                    
                ax.set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                ax.set_ylim(0, 1.1 * max(all_m_max_amplitudes))
            else:
                axes[channel_index].plot(m_x, [m_max_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_max_amplitudes)}\nAvg. M-max: {np.mean(m_max_amplitudes):.2f}mV\nStdev. M-Max: {np.std(m_max_amplitudes):.2f}mV\nAvg. Stim.: above {np.mean(m_max_thresholds):.2f} mV', xy=(m_x + 0.2, np.mean(m_max_amplitudes)), ha='left', va='center', color='black')
                axes[channel_index].errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_xticks([m_x])
                axes[channel_index].set_xticklabels(['M-response'])
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                axes[channel_index].set_ylim(0, 1.1 * max(all_m_max_amplitudes))

            # Append data to raw data dictionary
            datapoints = len(m_max_amplitudes)
            raw_data_dict['channel_index'].extend([channel_index]*datapoints)
            raw_data_dict['session_id'].extend(session_ids)
            raw_data_dict['m_max_threshold'].extend(m_max_thresholds)
            raw_data_dict['m_max_amplitude'].extend(m_max_amplitudes)

        # Set labels and title
        fig.suptitle('Average M-max')
        if num_channels == 1:
            ax.set_xlabel('Response Type')
            ax.set_ylabel(f'M-max (mV, {method})')
        else:
            fig.supxlabel('Response Type')
            fig.supylabel(f'M-max (mV, {method})')
        
        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index'], inplace=True)
        return raw_data_df        


