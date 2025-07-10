import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from typing import TYPE_CHECKING, List
from .base_plotter_pyqtgraph import BasePlotterPyQtGraph, UnableToPlotError

if TYPE_CHECKING:
    from monstim_signals.domain.dataset import Dataset
    from monstim_gui.widgets.plotting.plotting_widget import PlotPane


class DatasetPlotterPyQtGraph(BasePlotterPyQtGraph):
    """
    PyQtGraph-based plotter for Dataset data with interactive features.
    
    This class provides interactive plotting capabilities for EMG dataset data:
    - Real-time zooming and panning
    - Interactive latency window selection
    - Crosshair cursor for measurements
    - Multi-channel plotting support
    """
    
    def __init__(self, dataset: 'Dataset'):
        super().__init__(dataset)
        self.emg_object: 'Dataset' = dataset
        
    def plot_reflexCurves(self, channel_indices: List[int] = None, method : str = None, plot_legend : bool = True, 
                         relative_to_mmax : bool = False, manual_mmax : float | int | None =None,
                         interactive_cursor : bool = False, canvas : 'PlotPane'=None):
        """Plot average reflex curves for the dataset with interactive features, using domain Dataset object API."""
        try:
            if canvas is None:
                raise UnableToPlotError("Canvas is required for plotting.")

            self.clear_current_plots(canvas)

            # Get channel information from domain object
            if channel_indices is None:
                channel_indices = list(range(getattr(self.emg_object, 'n_channels', 1)))

            plot_items, layout = self.create_plot_layout(canvas, channel_indices)

            for plot_item, channel_idx in zip(plot_items, channel_indices):
                plot_item : 'pg.PlotItem'

                # Create plot item
                self.set_labels(plot_item=plot_item, title=f'{self.emg_object.channel_names[channel_idx]}',
                                y_label=f'Average Reflex Ampl. (mV{", rel. to M-max" if relative_to_mmax else ""})',
                                x_label='Stimulus Intensity (V)')
                plot_item.showGrid(True, True)
                self._plot_reflex_curves_data(plot_item, channel_idx, method, relative_to_mmax, manual_mmax)

            if interactive_cursor:
                self.add_synchronized_crosshairs(plot_items)

            # Add legend if requested, after all plotting is done
            if plot_legend and plot_items:
                if hasattr(plot_items[0], 'legend') and plot_items[0].legend is not None:
                    plot_items[0].removeItem(plot_items[0].legend)
                self.add_legend(plot_items[0])

            return self._extract_plot_data('reflexCurves', channel_indices, method, relative_to_mmax, manual_mmax)

        except Exception as e:
            raise UnableToPlotError(f"Error plotting reflex curves: {str(e)}")

    def _plot_reflex_curves_data(self, plot_item: pg.PlotItem, channel_idx, method, relative_to_mmax, manual_mmax):
        """Plot reflex curves data for a specific channel, robust to domain object return types."""
        try:
            # Get the data from the domain Dataset object
            for window in self.emg_object.latency_windows:
                window_reflex_data = self.emg_object.get_average_lw_reflex_curve(
                    method=method,
                    channel_index=channel_idx,
                    window=window
                )

                # Normalize to M-max if needed
                if relative_to_mmax and window_reflex_data['means'] is not None:
                    if manual_mmax is None:
                        mmax = self.emg_object.get_avg_m_max(
                            channel_index=channel_idx,
                            method=method
                        )
                    else:
                        mmax = manual_mmax
                    if mmax == 0:
                        raise ValueError("M-max cannot be zero when normalizing reflex curves.")
                    # Normalize means and stdevs by M-max
                    window_reflex_data['means'] = np.array(window_reflex_data['means']) / mmax
                    window_reflex_data['stdevs'] = np.array(window_reflex_data['stdevs']) / mmax

                color = self._convert_matplotlib_color(window.color)
                upper = np.array(window_reflex_data['means']) + np.array(window_reflex_data['stdevs'])
                lower = np.array(window_reflex_data['means']) - np.array(window_reflex_data['stdevs'])
                transparent_pen = pg.mkPen(color=color, width=1, style=QtCore.Qt.PenStyle.DotLine)
                upper_curve = plot_item.plot(window_reflex_data['voltages'], upper, pen=transparent_pen)
                lower_curve = plot_item.plot(window_reflex_data['voltages'], lower, pen=transparent_pen)
                if not hasattr(plot_item, '_fill_curves_refs'):
                    plot_item._fill_curves_refs = []
                plot_item._fill_curves_refs.extend([upper_curve, lower_curve])
                fill = pg.FillBetweenItem(curve1=upper_curve, curve2=lower_curve, brush=pg.mkBrush(color=color, alpha=50))
                plot_item.addItem(fill)
                # Plot mean line last so it appears on top and is visible
                plot_item.plot(
                    window_reflex_data['voltages'],
                    window_reflex_data['means'],
                    pen=pg.mkPen(color=color, width=2),
                    name=f'{window.name} Mean'
                )
        except Exception as e:
            print(f"Warning: Could not plot reflex curves for channel {channel_idx}: {e}")
    
    def plot_maxH(self, channel_indices: List[int] = None, method=None, relative_to_mmax=False, 
                  manual_mmax=None, max_stim_value=None, bin_margin=0, canvas=None):
        """Plot max H-reflex data with interactive features."""
        try:
            if canvas is None:
                raise UnableToPlotError("Canvas is required for plotting.")
            
            # Clear the canvas
            canvas.clear()
            
            # Get channel information
            if channel_indices is None:
                channel_indices = list(range(self.emg_object.n_channels))
            
            # Create subplot layout
            n_channels = len(channel_indices)
            cols = min(2, n_channels)
            rows = (n_channels + cols - 1) // cols
            
            plots = []
            for i, channel_idx in enumerate(channel_indices):
                row = i // cols
                col = i % cols
                
                # Create plot item
                plot_item = canvas.addPlot(row=row, col=col)
                plot_item.setLabel('left', f'Channel {channel_idx + 1}')
                plot_item.setLabel('bottom', 'Session')
                plot_item.showGrid(True, True)
                
                # Enable interactive features
                plot_item.setMouseEnabled(x=True, y=True)
                plot_item.enableAutoRange()
                
                # Add crosshair
                crosshair = self.add_crosshair(plot_item)
                
                # Plot max H data
                self._plot_max_h_data(plot_item, channel_idx, method, relative_to_mmax, 
                                    manual_mmax, max_stim_value, bin_margin)
                
                plots.append(plot_item)
            
            return self._extract_plot_data('maxH', channel_indices, method, relative_to_mmax, 
                                         manual_mmax, max_stim_value, bin_margin)
            
        except Exception as e:
            raise UnableToPlotError(f"Error plotting max H-reflex: {str(e)}")
    
    def _plot_max_h_data(self, plot_item, channel_idx, method, relative_to_mmax, 
                        manual_mmax, max_stim_value, bin_margin):
        """Plot max H-reflex data for a specific channel."""
        try:
            # Get the data from the dataset
            max_h_data = self.emg_object.get_max_h_reflex_data(
                channel_indices=[channel_idx],
                method=method,
                relative_to_mmax=relative_to_mmax,
                manual_mmax=manual_mmax,
                max_stim_value=max_stim_value,
                bin_margin=bin_margin
            )
            
            if max_h_data.empty:
                return
            
            # Create session labels for x-axis
            session_labels = [f"Session {i+1}" for i in range(len(max_h_data))]
            x_positions = np.arange(len(session_labels))
            
            # Plot M-wave data
            if 'M-wave' in max_h_data.columns:
                m_data = max_h_data['M-wave'].values
                plot_item.plot(x_positions, m_data,
                             pen=None, symbol='o', symbolSize=8, symbolBrush='red',
                             name='M-wave')
            
            # Plot H-reflex data
            if 'H-reflex' in max_h_data.columns:
                h_data = max_h_data['H-reflex'].values
                plot_item.plot(x_positions, h_data,
                             pen=None, symbol='s', symbolSize=8, symbolBrush='blue',
                             name='H-reflex')
            
            # Set x-axis labels
            plot_item.getAxis('bottom').setTicks([list(zip(x_positions, session_labels))])
            
        except Exception as e:
            print(f"Warning: Could not plot max H data for channel {channel_idx}: {e}")
    
    def plot_mmax(self, channel_indices: List[int] = None, method: str = None, canvas=None):
        """Plot M-max data with interactive features."""
        try:
            if canvas is None:
                raise UnableToPlotError("Canvas is required for plotting.")
            
            # Clear the canvas
            canvas.clear()
            
            # Get channel information
            if channel_indices is None:
                channel_indices = list(range(self.emg_object.n_channels))
            
            # Create subplot layout
            n_channels = len(channel_indices)
            cols = min(2, n_channels)
            rows = (n_channels + cols - 1) // cols
            
            plots = []
            for i, channel_idx in enumerate(channel_indices):
                row = i // cols
                col = i % cols
                
                # Create plot item
                plot_item = canvas.addPlot(row=row, col=col)
                plot_item.setLabel('left', f'Channel {channel_idx + 1}')
                plot_item.setLabel('bottom', 'Session')
                plot_item.showGrid(True, True)
                
                # Enable interactive features
                plot_item.setMouseEnabled(x=True, y=True)
                plot_item.enableAutoRange()
                
                # Add crosshair
                crosshair = self.add_crosshair(plot_item)
                
                # Plot M-max data
                self._plot_mmax_data(plot_item, channel_idx, method)
                
                plots.append(plot_item)
            
            return self._extract_plot_data('mmax', channel_indices, method)
            
        except Exception as e:
            raise UnableToPlotError(f"Error plotting M-max: {str(e)}")
    
    def _plot_mmax_data(self, plot_item, channel_idx, method):
        """Plot M-max data for a specific channel."""
        try:
            # Get the data from the dataset
            mmax_data = self.emg_object.get_mmax_data(
                channel_indices=[channel_idx],
                method=method
            )
            
            if mmax_data.empty:
                return
            
            # Create session labels for x-axis
            session_labels = [f"Session {i+1}" for i in range(len(mmax_data))]
            x_positions = np.arange(len(session_labels))
            
            # Plot M-max data
            mmax_values = mmax_data.iloc[:, 0].values  # Assuming first column is M-max
            plot_item.plot(x_positions, mmax_values,
                         pen=None, symbol='o', symbolSize=8, symbolBrush='green',
                         name='M-max')
            
            # Set x-axis labels
            plot_item.getAxis('bottom').setTicks([list(zip(x_positions, session_labels))])
            
        except Exception as e:
            print(f"Warning: Could not plot M-max data for channel {channel_idx}: {e}")
    
    def _extract_plot_data(self, plot_type, *args, **kwargs):
        """Extract raw data for the plot - placeholder for compatibility."""
        # This method should extract and return the raw data used in the plot
        # For now, return empty dict - can be implemented later as needed
        return {}
