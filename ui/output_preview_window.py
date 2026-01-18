"""
Output Preview Window for visualizing three-phase signal output.

This module provides a separate Toplevel window that displays:
1. INPUT: Position vs Time graph (2D line plot)
2. OUTPUTS: Three channel graphs (Neutral, Left, Right) showing
   amplitude over time with frequency encoded as color (green to red)
"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    # Only set backend if not already set
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass  # Backend already set, continue anyway
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: matplotlib not available for preview: {e}")
    MATPLOTLIB_AVAILABLE = False

from stim_math.threephase import ThreePhaseSignalGenerator
from funscript import Funscript
from processing.linear_mapping import apply_linear_response_curve


class OutputPreviewWindow(tk.Toplevel):
    """
    Output Preview Monitor window for visualizing three-phase signal output.

    Features:
    - Separate window (Toplevel) with its own file loading
    - INPUT: Position vs Time graph
    - OUTPUTS: 3 channel graphs (N, L, R) with amplitude and frequency-as-color
    - Timeline playback with Play/Pause controls
    """

    def __init__(
        self,
        parent: tk.Tk,
        config: Dict[str, Any],
        config_getter: Callable[[], Dict[str, Any]]
    ):
        """
        Initialize the output preview window.

        Args:
            parent: Parent tkinter window
            config: Current configuration dictionary
            config_getter: Callback to get latest config from main window
        """
        super().__init__(parent)

        self.title("Output Preview Monitor")
        self.geometry("900x800")
        self.minsize(700, 600)

        self.config = config
        self.config_getter = config_getter

        # Funscript data
        self.funscript: Optional[Funscript] = None
        self.funscript_path: Optional[Path] = None
        self.demo_file_var = tk.StringVar()

        # Playback state
        self.is_playing = False
        self.current_time_ms = 0
        self.playback_speed = 1.0
        self.total_duration_ms = 0
        self.update_interval_ms = 50  # 20 FPS
        self.view_window_seconds = 30.0  # Show 30 seconds of data at a time

        # Precomputed data for visualization
        self.time_points = None  # Time array in ms
        self.position_values = None  # Position values (0-1)
        self.speed_values = None  # Speed values (normalized 0-1)
        self.accel_values = None  # Acceleration values (normalized 0-1)
        self.alpha_values = None  # Alpha values (-1 to 1)
        self.beta_values = None  # Beta values (-1 to 1)
        self.n_amplitudes = None  # Neutral electrode amplitudes
        self.l_amplitudes = None  # Left electrode amplitudes
        self.r_amplitudes = None  # Right electrode amplitudes
        self.frequency_values = None  # Frequency values (normalized 0-1)

        # Motion axis data (E1-E4 with response curves applied)
        self.e1_values = None
        self.e2_values = None
        self.e3_values = None
        self.e4_values = None
        self.phase_shifted_positions = None  # Position with phase shift applied

        # Create custom colormap for frequency (green to red)
        # Only create if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            self.freq_cmap = LinearSegmentedColormap.from_list(
                'freq_cmap',
                [(0, 'green'), (0.5, 'yellow'), (1, 'red')]
            )
        else:
            self.freq_cmap = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the window UI components."""
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(
                self,
                text="Matplotlib is required for preview.\nPlease install: pip install matplotlib",
                font=('TkDefaultFont', 12)
            ).pack(expand=True)
            return

        # Main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill='both', expand=True)

        # File loader section (top)
        self._setup_file_loader(main_frame)

        # Playback controls FIRST at bottom - this ensures they're always visible
        # Pack with side='bottom' BEFORE the graphs so they get reserved space
        self._controls_frame = ttk.LabelFrame(main_frame, text="Playback Controls", padding="5")
        self._controls_frame.pack(side='bottom', fill='x', pady=(10, 0))

        # Graphs container fills remaining space (middle)
        graphs_frame = ttk.Frame(main_frame)
        graphs_frame.pack(fill='both', expand=True, pady=10)

        # Create matplotlib figure with 4 subplots
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.set_facecolor('#f0f0f0')

        # Create subplots with GridSpec for better control
        gs = self.fig.add_gridspec(
            4, 1,
            height_ratios=[1.5, 1, 1, 1],
            hspace=0.3
        )

        # INPUT: Position graph (larger)
        self.ax_position = self.fig.add_subplot(gs[0])
        self.ax_position.set_title('INPUT: Position vs Time', fontweight='bold', color='#2c3e50')
        self.ax_position.set_ylabel('Position')
        self.ax_position.set_xlim(0, 100)
        self.ax_position.set_ylim(0, 1)
        self.ax_position.grid(True, alpha=0.3)

        # OUTPUTS: Channel graphs
        self.ax_neutral = self.fig.add_subplot(gs[1])
        self.ax_neutral.set_title('OUTPUT: Neutral (N) Channel', fontweight='bold', color='#27ae60')
        self.ax_neutral.set_ylabel('Amplitude')
        self.ax_neutral.set_xlim(0, 100)
        self.ax_neutral.set_ylim(0, 1)
        self.ax_neutral.grid(True, alpha=0.3)

        self.ax_left = self.fig.add_subplot(gs[2])
        self.ax_left.set_title('OUTPUT: Left (L) Channel', fontweight='bold', color='#e74c3c')
        self.ax_left.set_ylabel('Amplitude')
        self.ax_left.set_xlim(0, 100)
        self.ax_left.set_ylim(0, 1)
        self.ax_left.grid(True, alpha=0.3)

        self.ax_right = self.fig.add_subplot(gs[3])
        self.ax_right.set_title('OUTPUT: Right (R) Channel', fontweight='bold', color='#3498db')
        self.ax_right.set_ylabel('Amplitude')
        self.ax_right.set_xlabel('Time (seconds)')
        self.ax_right.set_xlim(0, 100)
        self.ax_right.set_ylim(0, 1)
        self.ax_right.grid(True, alpha=0.3)

        # Add color legend for frequency
        self._add_frequency_legend()

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Playback indicator lines (vertical lines showing current time)
        self.playback_lines = []
        for ax in [self.ax_position, self.ax_neutral, self.ax_left, self.ax_right]:
            line = ax.axvline(x=0, color='red', linewidth=2, linestyle='--', alpha=0.7)
            self.playback_lines.append(line)

        # Playback controls
        self._setup_playback_controls(main_frame)

    def _setup_file_loader(self, parent):
        """Setup the file loader section."""
        file_frame = ttk.LabelFrame(parent, text="Demo Funscript", padding="5")
        file_frame.pack(fill='x', pady=(0, 10))

        ttk.Entry(
            file_frame,
            textvariable=self.demo_file_var,
            width=60
        ).pack(side='left', fill='x', expand=True, padx=(0, 10))

        ttk.Button(
            file_frame,
            text="Load Demo Funscript",
            command=self.browse_demo_funscript
        ).pack(side='left')

        ttk.Button(
            file_frame,
            text="Refresh from Settings",
            command=self.refresh_from_settings
        ).pack(side='left', padx=(10, 0))

    def _setup_playback_controls(self, parent):
        """Setup playback control panel."""
        # Use the pre-created controls frame (packed at bottom in _setup_ui)
        controls_frame = self._controls_frame

        # Top row: Play/Pause and timeline
        top_row = ttk.Frame(controls_frame)
        top_row.pack(fill='x', pady=(0, 5))

        # Play/Pause button
        self.play_button = ttk.Button(
            top_row,
            text="▶ Play",
            command=self.toggle_playback,
            width=10
        )
        self.play_button.pack(side='left', padx=(0, 10))

        # Stop button
        ttk.Button(
            top_row,
            text="⏹ Stop",
            command=self.stop_playback,
            width=10
        ).pack(side='left', padx=(0, 10))

        # Timeline slider
        self.timeline_var = tk.DoubleVar(value=0)
        self.timeline_slider = ttk.Scale(
            top_row,
            from_=0,
            to=100,
            variable=self.timeline_var,
            orient='horizontal',
            command=self._on_timeline_change
        )
        self.timeline_slider.pack(side='left', fill='x', expand=True, padx=(0, 10))

        # Time display
        self.time_label = ttk.Label(top_row, text="00:00.0 / 00:00.0", width=20)
        self.time_label.pack(side='left')

        # Bottom row: Speed control
        bottom_row = ttk.Frame(controls_frame)
        bottom_row.pack(fill='x')

        ttk.Label(bottom_row, text="Speed:").pack(side='left')

        self.speed_var = tk.StringVar(value="1.0x")
        speed_options = ["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"]
        speed_combo = ttk.Combobox(
            bottom_row,
            textvariable=self.speed_var,
            values=speed_options,
            width=8,
            state='readonly'
        )
        speed_combo.pack(side='left', padx=(5, 20))
        speed_combo.bind('<<ComboboxSelected>>', self._on_speed_change)

        # View window control
        ttk.Label(bottom_row, text="View Window:").pack(side='left')
        self.view_window_var = tk.StringVar(value="30s")
        view_options = ["10s", "30s", "60s", "120s", "Full"]
        view_combo = ttk.Combobox(
            bottom_row,
            textvariable=self.view_window_var,
            values=view_options,
            width=8,
            state='readonly'
        )
        view_combo.pack(side='left', padx=(5, 20))
        view_combo.bind('<<ComboboxSelected>>', self._on_view_window_change)

        # Frequency legend label
        ttk.Label(
            bottom_row,
            text="Freq: 🟢Low → 🔴High"
        ).pack(side='right')

    def _add_frequency_legend(self):
        """Add a color legend for frequency to the figure."""
        # Add a small colorbar-like legend
        # This is optional - the label in controls provides info too
        pass

    def browse_demo_funscript(self):
        """Open file dialog to select a demo funscript."""
        file_path = filedialog.askopenfilename(
            title="Select Demo Funscript",
            filetypes=[("Funscript files", "*.funscript"), ("All files", "*.*")]
        )
        if file_path:
            self.load_demo_funscript(file_path)

    def load_demo_funscript(self, path: str):
        """
        Load a funscript file for preview.

        Args:
            path: Path to the funscript file
        """
        try:
            self.funscript_path = Path(path)
            self.funscript = Funscript.from_file(self.funscript_path)
            self.demo_file_var.set(str(self.funscript_path))

            # Compute preview data
            self._compute_preview_data()

            # Update display
            self._update_graphs()

            # Reset playback
            self.stop_playback()

        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load funscript: {e}")

    def refresh_from_settings(self):
        """Refresh preview based on current settings from main window."""
        if self.config_getter:
            self.config = self.config_getter()

        if self.funscript is not None:
            # Get key settings for status display
            general = self.config.get('general', {})
            freq = self.config.get('frequency', {})
            ab = self.config.get('alpha_beta_generation', {})
            pulse = self.config.get('pulse', {})
            volume = self.config.get('volume', {})
            advanced = self.config.get('advanced', {})
            pos_axes = self.config.get('positional_axes', {})

            rest_level = general.get('rest_level', 0.4)
            freq_min = freq.get('pulse_freq_min', 0.4)
            freq_max = freq.get('pulse_freq_max', 0.95)
            vol_ramp_ratio = volume.get('volume_ramp_combine_ratio', 20.0)
            vol_inv = advanced.get('enable_volume_inversion', False)
            freq_inv = advanced.get('enable_frequency_inversion', False)

            # Motion axis settings
            mode = pos_axes.get('mode', 'motion_axis')
            phase_shift = pos_axes.get('phase_shift', {})
            phase_enabled = phase_shift.get('enabled', False)
            delay_pct = phase_shift.get('delay_percentage', 10.0)

            # Count enabled axes
            enabled_axes = []
            for axis in ['e1', 'e2', 'e3', 'e4']:
                if pos_axes.get(axis, {}).get('enabled', False):
                    enabled_axes.append(axis.upper())

            # Update window title to show current settings
            flags_str = ""
            if vol_inv:
                flags_str += " VolInv"
            if freq_inv:
                flags_str += " FreqInv"
            if phase_enabled:
                flags_str += f" Phase:{delay_pct:.0f}%"

            axes_str = ",".join(enabled_axes) if enabled_axes else "None"
            self.title(f"Preview - Axes:{axes_str} Rest:{rest_level:.2f} Freq:{freq_min:.2f}-{freq_max:.2f}{flags_str}")

            self._compute_preview_data()
            self._update_graphs()
        else:
            tk.messagebox.showinfo("Info", "Load a funscript first to see changes")

    def _compute_preview_data(self):
        """Compute all preview data from the loaded funscript."""
        if self.funscript is None:
            return

        # Get funscript data
        # Note: Funscript class already converts:
        #   - x from milliseconds to SECONDS (at / 1000)
        #   - y from 0-100 to 0-1 (pos * 0.01)
        times_s = self.funscript.x  # Time in seconds (already converted)
        positions = self.funscript.y  # Position 0-1 (already converted)

        if len(times_s) < 2:
            return

        # Convert to milliseconds for internal tracking
        self.total_duration_ms = int(times_s[-1] * 1000)

        # Create interpolated time points for smooth visualization (in ms)
        num_points = min(2000, max(200, len(times_s) * 3))
        self.time_points = np.linspace(0, self.total_duration_ms, num_points)

        # Interpolate position values (convert time_points to seconds for interp)
        # This is the RAW position from the input file (shown in INPUT graph)
        self.position_values = np.interp(self.time_points / 1000.0, times_s, positions)

        # Compute speed values first (needed by other computations)
        self._compute_speed_values()

        # Apply motion axis transformations based on mode
        # This creates self.transformed_positions which is used for N/L/R computation
        self._apply_motion_axis_mode()

        # Convert transformed position to alpha/beta using algorithm settings
        self._compute_alpha_beta()

        # Compute electrode amplitudes using volume/pulse settings
        self._compute_electrode_amplitudes()

        # Compute frequency values using frequency settings
        self._compute_frequency_values()

        # Update timeline slider range
        self.timeline_slider.configure(to=self.total_duration_ms)

    def _compute_speed_values(self):
        """Compute speed values from position for use in other calculations."""
        if self.position_values is None or self.time_points is None:
            self.speed_values = None
            self.accel_values = None
            return

        config = self.config_getter() if self.config_getter else self.config
        general_config = config.get('general', {})
        speed_config = config.get('speed', {})

        # Get speed calculation parameters
        speed_window = general_config.get('speed_window_size', 5)
        accel_window = general_config.get('accel_window_size', 3)
        interp_interval = speed_config.get('interpolation_interval', 0.1)
        normalization_method = speed_config.get('normalization_method', 'max')

        # Calculate speed (derivative of position)
        # Use interpolation_interval to determine step size for derivative
        dt = np.diff(self.time_points / 1000.0)  # Time differences in seconds
        dt = np.append(dt, dt[-1])  # Pad to match length

        # Resample based on interpolation_interval if needed
        # Higher interval = smoother speed calculation
        step_samples = max(1, int(interp_interval * 1000 / self.update_interval_ms))

        pos_diff = np.diff(self.position_values)
        pos_diff = np.append(pos_diff, pos_diff[-1])

        raw_speed = np.abs(pos_diff / np.maximum(dt, 0.001))

        # Apply smoothing window (speed_window_size effect)
        if speed_window > 1 and len(raw_speed) > speed_window:
            kernel = np.ones(speed_window) / speed_window
            self.speed_values = np.convolve(raw_speed, kernel, mode='same')
        else:
            self.speed_values = raw_speed

        # Compute acceleration (derivative of speed) using accel_window_size
        speed_diff = np.diff(self.speed_values)
        speed_diff = np.append(speed_diff, speed_diff[-1])
        raw_accel = speed_diff / np.maximum(dt, 0.001)

        # Apply acceleration smoothing window (accel_window_size effect)
        if accel_window > 1 and len(raw_accel) > accel_window:
            kernel = np.ones(accel_window) / accel_window
            self.accel_values = np.convolve(raw_accel, kernel, mode='same')
        else:
            self.accel_values = raw_accel

        # Normalize speed based on normalization_method
        if normalization_method == 'max':
            max_speed = np.max(self.speed_values) if len(self.speed_values) > 0 else 1.0
        elif normalization_method == 'percentile':
            max_speed = np.percentile(self.speed_values, 95) if len(self.speed_values) > 0 else 1.0
        else:  # 'mean' or default
            max_speed = np.mean(self.speed_values) * 2 if len(self.speed_values) > 0 else 1.0

        max_speed = max(max_speed, 0.001)
        self.speed_values = np.clip(self.speed_values / max_speed, 0, 1)

        # Normalize acceleration to 0-1 range
        max_accel = np.percentile(np.abs(self.accel_values), 95) if len(self.accel_values) > 0 else 1.0
        max_accel = max(max_accel, 0.001)
        self.accel_values = np.clip(np.abs(self.accel_values) / max_accel, 0, 1)

    def _apply_motion_axis_mode(self):
        """
        Apply motion axis transformations based on the selected mode.

        In 'legacy' mode: use raw position values directly
        In 'motion_axis' mode: apply phase shift and E1-E4 response curves

        The result is stored in self.transformed_positions which is used
        for computing N/L/R electrode amplitudes.
        """
        if self.position_values is None:
            self.transformed_positions = None
            return

        config = self.config_getter() if self.config_getter else self.config
        pos_config = config.get('positional_axes', {})

        # Get mode
        mode = pos_config.get('mode', 'motion_axis')

        if mode == 'legacy':
            # Legacy mode: use raw position values directly
            self.transformed_positions = self.position_values.copy()
            # Clear E1-E4 values since we're not using them
            self.e1_values = None
            self.e2_values = None
            self.e3_values = None
            self.e4_values = None
            self.phase_shifted_positions = None
        else:
            # Motion axis mode: apply phase shift and response curves
            # Get phase shift settings
            phase_shift_config = pos_config.get('phase_shift', {})
            phase_shift_enabled = phase_shift_config.get('enabled', False)
            delay_percentage = phase_shift_config.get('delay_percentage', 10.0)
            min_segment_duration = phase_shift_config.get('min_segment_duration', 0.25)

            # Start with original position values
            base_positions = self.position_values.copy()

            # Apply phase shift if enabled
            if phase_shift_enabled and delay_percentage > 0:
                self.phase_shifted_positions = self._apply_phase_shift(
                    base_positions,
                    delay_percentage,
                    min_segment_duration
                )
            else:
                self.phase_shifted_positions = base_positions

            # Compute E1-E4 values by applying response curves
            enabled_axes_values = []
            for axis_name in ['e1', 'e2', 'e3', 'e4']:
                axis_config = pos_config.get(axis_name, {})
                enabled = axis_config.get('enabled', False)

                if enabled:
                    curve_config = axis_config.get('curve', {})
                    control_points = curve_config.get('control_points', [(0.0, 0.0), (1.0, 1.0)])

                    # Apply response curve to each position value
                    axis_values = np.array([
                        apply_linear_response_curve(pos, control_points)
                        for pos in self.phase_shifted_positions
                    ])

                    # Store the computed values
                    setattr(self, f'{axis_name}_values', axis_values)
                    enabled_axes_values.append(axis_values)
                else:
                    # Disabled axis - set to None
                    setattr(self, f'{axis_name}_values', None)

            # Combine enabled axes to create transformed positions
            # The transformed position affects how N/L/R are computed
            if enabled_axes_values:
                # Average all enabled axis values to create the transformed position
                # This represents the combined effect of all motion axes
                self.transformed_positions = np.mean(enabled_axes_values, axis=0)
            else:
                # No axes enabled, use phase-shifted positions
                self.transformed_positions = self.phase_shifted_positions

    def _compute_alpha_beta(self):
        """Convert position values to alpha/beta coordinates using all relevant settings."""
        # Use transformed positions (affected by motion axis mode)
        if self.transformed_positions is None:
            if self.position_values is None:
                return
            self.transformed_positions = self.position_values.copy()

        # Get conversion settings from config
        config = self.config_getter() if self.config_getter else self.config
        ab_config = config.get('alpha_beta_generation', {})
        pulse_config = config.get('pulse', {})

        algorithm = ab_config.get('algorithm', 'top-right-left')
        min_distance = ab_config.get('min_distance_from_center', 0.1)
        speed_threshold = ab_config.get('speed_threshold_percent', 50) / 100.0
        direction_change_prob = ab_config.get('direction_change_probability', 0.1)

        # Beta mirror threshold affects how beta is computed
        beta_mirror_threshold = pulse_config.get('beta_mirror_threshold', 0.5)

        # Use transformed positions (affected by motion axis mode settings)
        pos = self.transformed_positions

        # Scale position to create movement from center outward
        # Use speed to modulate radius (faster = further out)
        if self.speed_values is not None:
            # Speed above threshold pushes position outward more
            speed_factor = np.where(self.speed_values > speed_threshold,
                                   1.0 + (self.speed_values - speed_threshold) * 0.5,
                                   1.0)
            effective_pos = np.clip(pos * speed_factor, 0, 1)
        else:
            effective_pos = pos

        radius = min_distance + (1 - min_distance) * effective_pos

        # Create circular/arc motion based on algorithm
        if algorithm == 'circular':
            # Full circular motion
            angle = pos * np.pi  # 0 to 180 degrees
        elif algorithm == 'top-left-right':
            # Counter-clockwise arc
            angle = np.pi/2 + pos * np.pi  # 90 to 270 degrees
        elif algorithm == 'top-right-left':
            # Clockwise arc
            angle = np.pi/2 - pos * np.pi  # 90 to -90 degrees
        else:
            # Default: vertical motion
            angle = np.pi/2

        self.alpha_values = radius * np.cos(angle)
        self.beta_values = radius * np.sin(angle)

        # Apply direction_change_probability - adds variation to the path
        # Higher probability = more lateral movement/variation in the signal
        # This simulates random direction changes that occur in the actual algorithm
        if direction_change_prob > 0 and self.accel_values is not None:
            # Use acceleration to determine where direction changes might occur
            # direction_change_prob scales how much variation is added
            variation_strength = direction_change_prob * 0.5  # Scale to reasonable range

            # Add variation to alpha based on acceleration (simulates direction changes)
            # More acceleration = more likely to have direction variation
            alpha_variation = self.accel_values * variation_strength * np.sin(pos * np.pi * 4)
            beta_variation = self.accel_values * variation_strength * np.cos(pos * np.pi * 4)

            self.alpha_values = self.alpha_values + alpha_variation
            self.beta_values = self.beta_values + beta_variation

        # Apply beta mirror threshold - mirror beta when below threshold
        if beta_mirror_threshold > 0:
            mirror_mask = np.abs(self.beta_values) < beta_mirror_threshold
            # This simulates the beta mirroring effect
            self.beta_values = np.where(mirror_mask,
                                        self.beta_values * (1 + beta_mirror_threshold),
                                        self.beta_values)

        # Clamp to unit circle
        r = np.sqrt(self.alpha_values**2 + self.beta_values**2)
        mask = r > 1
        self.alpha_values = np.where(mask, self.alpha_values / r, self.alpha_values)
        self.beta_values = np.where(mask, self.beta_values / r, self.beta_values)

    def _compute_electrode_amplitudes(self):
        """Compute electrode amplitudes from alpha/beta values, applying all volume/pulse settings."""
        if self.alpha_values is None or self.beta_values is None:
            return

        config = self.config_getter() if self.config_getter else self.config
        general_config = config.get('general', {})
        volume_config = config.get('volume', {})
        pulse_config = config.get('pulse', {})
        advanced_config = config.get('advanced', {})

        # Get settings
        rest_level = general_config.get('rest_level', 0.4)
        ramp_up_duration = general_config.get('ramp_up_duration_after_rest', 1.0)

        # Volume settings
        volume_ramp_ratio = volume_config.get('volume_ramp_combine_ratio', 20.0)
        ramp_percent_per_hour = volume_config.get('ramp_percent_per_hour', 15)

        # Pulse width settings (affects perceived intensity)
        pulse_width_min = pulse_config.get('pulse_width_min', 0.1)
        pulse_width_max = pulse_config.get('pulse_width_max', 0.45)
        pulse_width_ratio = pulse_config.get('pulse_width_combine_ratio', 3)

        # Pulse rise settings
        pulse_rise_min = pulse_config.get('pulse_rise_min', 0.0)
        pulse_rise_max = pulse_config.get('pulse_rise_max', 0.8)
        pulse_rise_ratio = pulse_config.get('pulse_rise_combine_ratio', 2)

        # Advanced inversion settings
        enable_volume_inversion = advanced_config.get('enable_volume_inversion', False)

        n_points = len(self.alpha_values)
        self.n_amplitudes = np.zeros(n_points)
        self.l_amplitudes = np.zeros(n_points)
        self.r_amplitudes = np.zeros(n_points)

        # Compute base amplitude for each point
        for i in range(n_points):
            n, l, r = ThreePhaseSignalGenerator.electrode_amplitude(
                self.alpha_values[i],
                self.beta_values[i]
            )
            self.n_amplitudes[i] = float(n)
            self.l_amplitudes[i] = float(l)
            self.r_amplitudes[i] = float(r)

        # Normalize to 0-1 range
        max_amp = max(
            np.max(self.n_amplitudes),
            np.max(self.l_amplitudes),
            np.max(self.r_amplitudes),
            0.001
        )
        self.n_amplitudes /= max_amp
        self.l_amplitudes /= max_amp
        self.r_amplitudes /= max_amp

        # Compute pulse width factor (higher position = wider pulse = stronger feel)
        pulse_width_range = pulse_width_max - pulse_width_min
        pulse_width_factor = pulse_width_min + pulse_width_range * self.position_values

        # Compute pulse rise factor
        pulse_rise_range = pulse_rise_max - pulse_rise_min
        if self.speed_values is not None:
            # Pulse rise affected by speed
            pulse_rise_factor = pulse_rise_min + pulse_rise_range * self.speed_values
        else:
            pulse_rise_factor = pulse_rise_min + pulse_rise_range * self.position_values

        # Combine factors to modulate amplitude
        # pulse_width_ratio determines how much pulse width affects amplitude
        width_weight = pulse_width_ratio / (pulse_width_ratio + 1)
        rise_weight = pulse_rise_ratio / (pulse_rise_ratio + 1)

        modulation = (1 - width_weight) + width_weight * pulse_width_factor
        modulation *= (1 - rise_weight) + rise_weight * pulse_rise_factor

        # Apply modulation to amplitudes
        self.n_amplitudes *= modulation
        self.l_amplitudes *= modulation
        self.r_amplitudes *= modulation

        # Apply ramp_up_duration_after_rest - gradual volume increase after rest periods
        # Detect rest periods (low position values) and apply ramp-up after them
        if ramp_up_duration > 0:
            rest_threshold = rest_level * 1.2  # Consider positions near rest_level as "rest"
            ramp_samples = int(ramp_up_duration * 1000 / self.update_interval_ms)

            # Find rest periods and apply ramp-up
            is_resting = self.position_values < rest_threshold
            ramp_factor = np.ones(len(self.position_values))

            in_ramp = False
            ramp_counter = 0
            for i in range(len(self.position_values)):
                if is_resting[i]:
                    in_ramp = True
                    ramp_counter = 0
                elif in_ramp:
                    ramp_counter += 1
                    # Gradual ramp from 0.5 to 1.0 over ramp_up_duration
                    ramp_progress = min(1.0, ramp_counter / max(ramp_samples, 1))
                    ramp_factor[i] = 0.5 + 0.5 * ramp_progress
                    if ramp_counter >= ramp_samples:
                        in_ramp = False

            self.n_amplitudes *= ramp_factor
            self.l_amplitudes *= ramp_factor
            self.r_amplitudes *= ramp_factor

        # Apply ramp_percent_per_hour - gradual intensity increase over time
        # This simulates the session-long volume ramp
        if ramp_percent_per_hour > 0:
            total_duration_hours = self.total_duration_ms / (1000 * 60 * 60)
            time_normalized = self.time_points / self.total_duration_ms  # 0 to 1

            # Calculate ramp multiplier: starts at (1 - total_ramp/2) and ends at (1 + total_ramp/2)
            total_ramp = (ramp_percent_per_hour / 100.0) * total_duration_hours
            ramp_multiplier = (1 - total_ramp/2) + total_ramp * time_normalized

            # Clamp to reasonable range
            ramp_multiplier = np.clip(ramp_multiplier, 0.5, 1.5)

            self.n_amplitudes *= ramp_multiplier
            self.l_amplitudes *= ramp_multiplier
            self.r_amplitudes *= ramp_multiplier

        # Apply volume ramp ratio - this determines how much position affects volume
        # Higher ratio = position has more influence on volume (more dynamic range)
        # Lower ratio = more constant volume regardless of position
        # volume_ramp_ratio range is 10-40, normalize to create a weight factor
        volume_weight = volume_ramp_ratio / 40.0  # Normalized 0.25-1.0

        # Position-based volume scaling (higher position = higher volume)
        position_volume = self.position_values ** (1.0 / max(volume_weight * 2, 0.5))

        # Apply rest level as baseline, with volume_ramp_ratio affecting the dynamic range
        # High ratio (40): full dynamic range from rest_level to 1.0
        # Low ratio (10): compressed dynamic range, stays closer to middle
        dynamic_range = (1 - rest_level) * volume_weight

        self.n_amplitudes = rest_level + dynamic_range * self.n_amplitudes * position_volume
        self.l_amplitudes = rest_level + dynamic_range * self.l_amplitudes * position_volume
        self.r_amplitudes = rest_level + dynamic_range * self.r_amplitudes * position_volume

        # Apply volume inversion if enabled
        if enable_volume_inversion:
            self.n_amplitudes = 1.0 - self.n_amplitudes
            self.l_amplitudes = 1.0 - self.l_amplitudes
            self.r_amplitudes = 1.0 - self.r_amplitudes

        # Clamp to 0-1
        self.n_amplitudes = np.clip(self.n_amplitudes, 0, 1)
        self.l_amplitudes = np.clip(self.l_amplitudes, 0, 1)
        self.r_amplitudes = np.clip(self.r_amplitudes, 0, 1)

    def _compute_frequency_values(self):
        """Compute frequency values based on ALL frequency config settings."""
        if self.time_points is None or self.position_values is None:
            return

        config = self.config_getter() if self.config_getter else self.config
        freq_config = config.get('frequency', {})
        advanced_config = config.get('advanced', {})

        # Get ALL frequency parameters from config
        freq_min = freq_config.get('pulse_freq_min', 0.4)
        freq_max = freq_config.get('pulse_freq_max', 0.95)
        freq_ramp_ratio = freq_config.get('frequency_ramp_combine_ratio', 2)
        freq_combine_ratio = freq_config.get('pulse_frequency_combine_ratio', 3)

        # Advanced settings
        enable_freq_inversion = advanced_config.get('enable_frequency_inversion', False)
        enable_pulse_freq_inversion = advanced_config.get('enable_pulse_frequency_inversion', False)

        # Use precomputed speed values if available
        if self.speed_values is not None:
            speed_normalized = self.speed_values
        else:
            # Compute speed locally
            dt = np.diff(self.time_points / 1000.0)
            dt = np.append(dt, dt[-1])
            pos_diff = np.diff(self.position_values)
            pos_diff = np.append(pos_diff, pos_diff[-1])
            speed = np.abs(pos_diff / np.maximum(dt, 0.001))
            max_speed = np.percentile(speed, 95) if len(speed) > 0 else 1.0
            max_speed = max(max_speed, 0.001)
            speed_normalized = np.clip(speed / max_speed, 0, 1)

        # Combine position and speed for frequency
        # freq_combine_ratio determines position vs speed weighting
        position_weight = 1.0 / (freq_combine_ratio + 1)
        speed_weight = freq_combine_ratio / (freq_combine_ratio + 1)

        combined = position_weight * self.position_values + speed_weight * speed_normalized

        # Apply frequency ramp ratio (affects how quickly frequency changes)
        # Higher ratio = more aggressive frequency response
        combined = np.power(combined, 1.0 / freq_ramp_ratio)

        # Map to frequency range
        freq_range = freq_max - freq_min
        frequency_raw = freq_min + freq_range * combined

        # Apply inversions if enabled
        if enable_freq_inversion:
            frequency_raw = freq_max - (frequency_raw - freq_min)

        if enable_pulse_freq_inversion:
            frequency_raw = freq_max - (frequency_raw - freq_min)

        # Normalize to 0-1 for colormap display
        # 0 = freq_min (green), 1 = freq_max (red)
        self.frequency_values = np.clip((frequency_raw - freq_min) / max(freq_range, 0.001), 0, 1)

    def _apply_phase_shift(self, positions, delay_percentage, min_segment_duration):
        """
        Apply phase shift to position values.

        This shifts the timing of position changes to create a delayed response,
        simulating the phase shift effect used in the actual output generation.

        Args:
            positions: Original position values
            delay_percentage: Percentage of segment duration to delay (0-100)
            min_segment_duration: Minimum time between extremes in seconds

        Returns:
            Phase-shifted position values
        """
        if delay_percentage <= 0:
            return positions.copy()

        # Convert delay percentage to fraction
        delay_fraction = delay_percentage / 100.0

        # Find local extrema (peaks and valleys) to identify segments
        # A segment is the time between two consecutive extrema
        extrema_indices = []

        for i in range(1, len(positions) - 1):
            # Check if this is a local maximum or minimum
            is_max = positions[i] > positions[i-1] and positions[i] > positions[i+1]
            is_min = positions[i] < positions[i-1] and positions[i] < positions[i+1]

            if is_max or is_min:
                # Check minimum segment duration
                if len(extrema_indices) > 0:
                    time_diff = (self.time_points[i] - self.time_points[extrema_indices[-1]]) / 1000.0
                    if time_diff >= min_segment_duration:
                        extrema_indices.append(i)
                else:
                    extrema_indices.append(i)

        if len(extrema_indices) < 2:
            return positions.copy()

        # Apply phase shift by delaying the transition between extrema
        shifted = positions.copy()

        for seg_idx in range(len(extrema_indices) - 1):
            start_idx = extrema_indices[seg_idx]
            end_idx = extrema_indices[seg_idx + 1]
            segment_length = end_idx - start_idx

            # Calculate delay in samples
            delay_samples = int(segment_length * delay_fraction)

            if delay_samples > 0 and delay_samples < segment_length:
                # Shift the segment by holding the start value longer
                for i in range(start_idx, min(start_idx + delay_samples, end_idx)):
                    shifted[i] = positions[start_idx]

                # Compress the remaining transition
                remaining_length = segment_length - delay_samples
                if remaining_length > 0:
                    for i in range(delay_samples, segment_length):
                        target_idx = start_idx + i
                        if target_idx < end_idx:
                            # Map from compressed range to original range
                            source_progress = (i - delay_samples) / remaining_length
                            source_idx = start_idx + int(source_progress * segment_length)
                            source_idx = min(source_idx, end_idx - 1)
                            shifted[target_idx] = positions[source_idx]

        return shifted

    def _get_view_window(self):
        """Calculate the current view window based on playback position."""
        current_time_s = self.current_time_ms / 1000.0
        total_duration_s = self.total_duration_ms / 1000.0
        half_window = self.view_window_seconds / 2.0

        # Center the window on current time, but clamp to valid range
        window_start = current_time_s - half_window
        window_end = current_time_s + half_window

        # Adjust if window extends beyond bounds
        if window_start < 0:
            window_start = 0
            window_end = min(self.view_window_seconds, total_duration_s)
        elif window_end > total_duration_s:
            window_end = total_duration_s
            window_start = max(0, total_duration_s - self.view_window_seconds)

        return window_start, window_end

    def _update_graphs(self):
        """Update all graph displays with windowed view."""
        if not MATPLOTLIB_AVAILABLE or self.time_points is None:
            return

        time_seconds = self.time_points / 1000.0  # Convert to seconds

        # Get view window
        view_start, view_end = self._get_view_window()

        # Clear and redraw position graph - INPUT shows only raw funscript data
        self.ax_position.clear()
        self.ax_position.set_title('INPUT: Position vs Time', fontweight='bold', color='#2c3e50')
        self.ax_position.set_ylabel('Position')
        self.ax_position.set_xlabel('')
        self.ax_position.set_xlim(view_start, view_end)
        self.ax_position.set_ylim(0, 1)
        self.ax_position.grid(True, alpha=0.3)

        # Draw original position from input file only
        self.ax_position.fill_between(time_seconds, self.position_values, alpha=0.3, color='#3498db')
        self.ax_position.plot(time_seconds, self.position_values, color='#2c3e50', linewidth=1.5)

        # Draw colored output graphs with same view window
        self._draw_colored_graph(self.ax_neutral, time_seconds, self.n_amplitudes,
                                'OUTPUT: Neutral (N) Channel', '#27ae60', view_start, view_end)
        self._draw_colored_graph(self.ax_left, time_seconds, self.l_amplitudes,
                                'OUTPUT: Left (L) Channel', '#e74c3c', view_start, view_end)
        self._draw_colored_graph(self.ax_right, time_seconds, self.r_amplitudes,
                                'OUTPUT: Right (R) Channel', '#3498db', view_start, view_end)

        self.ax_right.set_xlabel('Time (seconds)')

        # Recreate playback lines at current position
        current_time_s = self.current_time_ms / 1000.0
        self.playback_lines = []
        for ax in [self.ax_position, self.ax_neutral, self.ax_left, self.ax_right]:
            line = ax.axvline(x=current_time_s, color='red', linewidth=2, linestyle='--', alpha=0.7)
            self.playback_lines.append(line)

        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_colored_graph(self, ax, time_seconds, amplitudes, title, base_color, view_start=None, view_end=None):
        """
        Draw a graph with amplitude colored by frequency.

        Args:
            ax: Matplotlib axis
            time_seconds: Time array in seconds
            amplitudes: Amplitude array (0-1)
            title: Graph title
            base_color: Fallback base color
            view_start: Start of view window in seconds (optional)
            view_end: End of view window in seconds (optional)
        """
        ax.clear()
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Amplitude')

        # Set view window
        if view_start is not None and view_end is not None:
            ax.set_xlim(view_start, view_end)
        else:
            ax.set_xlim(0, time_seconds[-1])

        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        if self.frequency_values is None or len(self.frequency_values) == 0:
            ax.plot(time_seconds, amplitudes, color=base_color, linewidth=1.5)
            return

        # Create colored line segments based on frequency
        points = np.array([time_seconds, amplitudes]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create LineCollection with colors based on frequency
        lc = LineCollection(segments, cmap=self.freq_cmap, norm=plt.Normalize(0, 1))
        lc.set_array(self.frequency_values[:-1])
        lc.set_linewidth(2)
        ax.add_collection(lc)

        # Add filled area with gradient effect (simplified - use average color)
        avg_freq = np.mean(self.frequency_values)
        fill_color = self.freq_cmap(avg_freq)
        ax.fill_between(time_seconds, amplitudes, alpha=0.2, color=fill_color)

    def toggle_playback(self):
        """Toggle play/pause state."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Start playback."""
        if self.funscript is None:
            return

        self.is_playing = True
        self.play_button.configure(text="⏸ Pause")
        self._playback_tick()

    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.play_button.configure(text="▶ Play")

    def stop_playback(self):
        """Stop playback and reset to beginning."""
        self.is_playing = False
        self.current_time_ms = 0
        self.timeline_var.set(0)
        self.play_button.configure(text="▶ Play")
        self._update_playback_position(0)
        self._update_time_label()

    def seek(self, time_ms: int):
        """
        Seek to a specific time.

        Args:
            time_ms: Time in milliseconds
        """
        self.current_time_ms = max(0, min(time_ms, self.total_duration_ms))
        self.timeline_var.set(self.current_time_ms)
        self._update_playback_position(self.current_time_ms)
        self._update_time_label()

    def _on_timeline_change(self, value):
        """Handle timeline slider change."""
        if not self.is_playing:
            time_ms = float(value)
            self.current_time_ms = int(time_ms)
            self._update_playback_position(self.current_time_ms)
            self._update_time_label()

    def _on_speed_change(self, event):
        """Handle playback speed change."""
        speed_str = self.speed_var.get()
        self.playback_speed = float(speed_str.replace('x', ''))

    def _on_view_window_change(self, event):
        """Handle view window size change."""
        view_str = self.view_window_var.get()
        if view_str == "Full":
            # Show entire duration
            self.view_window_seconds = self.total_duration_ms / 1000.0 + 1
        else:
            # Parse seconds from string like "30s"
            self.view_window_seconds = float(view_str.replace('s', ''))

        # Update display immediately
        self._update_playback_position(self.current_time_ms)

    def _playback_tick(self):
        """Advance playback by one tick."""
        if not self.is_playing:
            return

        # Advance time
        self.current_time_ms += int(self.update_interval_ms * self.playback_speed)

        # Check if reached end
        if self.current_time_ms >= self.total_duration_ms:
            self.current_time_ms = self.total_duration_ms
            self.pause()

        # Update display
        self.timeline_var.set(self.current_time_ms)
        self._update_playback_position(self.current_time_ms)
        self._update_time_label()

        # Schedule next tick
        if self.is_playing:
            self.after(self.update_interval_ms, self._playback_tick)

    def _update_playback_position(self, time_ms: int):
        """Update the playback indicator lines and view window."""
        if not MATPLOTLIB_AVAILABLE:
            return

        time_seconds = time_ms / 1000.0

        # Update view window for all axes
        view_start, view_end = self._get_view_window()
        for ax in [self.ax_position, self.ax_neutral, self.ax_left, self.ax_right]:
            ax.set_xlim(view_start, view_end)

        # Update playback lines
        for line in self.playback_lines:
            line.set_xdata([time_seconds, time_seconds])

        self.canvas.draw_idle()

    def _update_time_label(self):
        """Update the time display label."""
        current_s = self.current_time_ms / 1000.0
        total_s = self.total_duration_ms / 1000.0

        current_min = int(current_s // 60)
        current_sec = current_s % 60

        total_min = int(total_s // 60)
        total_sec = total_s % 60

        self.time_label.configure(
            text=f"{current_min:02d}:{current_sec:04.1f} / {total_min:02d}:{total_sec:04.1f}"
        )

    def update_from_config(self, config: Dict[str, Any]):
        """
        Update preview based on configuration changes.

        Args:
            config: Updated configuration dictionary
        """
        self.config = config
        if self.funscript is not None:
            self._compute_preview_data()
            self._update_graphs()
