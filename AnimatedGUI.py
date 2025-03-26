import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PerformanceAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1400x900")
        self.root.configure(bg="#1f1f2b")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.show_welcome_page()

    def show_welcome_page(self):
        self.welcome_frame = ctk.CTkFrame(self.root, fg_color="#1f1f2b")
        self.welcome_frame.pack(fill="both", expand=True)

        title_label = ctk.CTkLabel(
            self.welcome_frame,
            text="AI-Powered System Performance Analyzer",
            font=("Inter", 36, "bold"),
            text_color="white"
        )
        title_label.place(relx=0.5, rely=0.35, anchor="center")

        welcome_label = ctk.CTkLabel(
            self.welcome_frame,
            text="Welcome Master Ojas",
            font=("Inter", 24),
            text_color="white"
        )
        welcome_label.place(relx=0.5, rely=0.45, anchor="center")

        self.enter_button = ctk.CTkButton(
            self.welcome_frame,
            text="Enter",
            command=self.show_main_page,
            corner_radius=10,
            font=("Inter", 16, "bold"),
            width=200,
            height=50,
            hover_color="#0056b3"
        )
        self.enter_button.place(relx=0.5, rely=0.55, anchor="center")

    def show_main_page(self):
        self.welcome_frame.destroy()
        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)

        self.data = None
        self.is_tracking = False
        self.tracking_thread = None
        self.performance_data = {
            'timestamps': [], 'cpu_usage': [], 'memory_usage': [],
            'disk_read': [], 'disk_write': [], 'network_sent': [], 'network_recv': []
        }
        self.create_main_ui()

    def create_main_ui(self):
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 10))

        header = ctk.CTkLabel(
            header_frame,
            text="AI-Powered System Performance Analyzer",
            font=("Inter", 24, "bold"),
            text_color="white"
        )
        header.pack(pady=15)

        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.control_panel = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.control_panel.pack(side="left", fill="y", padx=(0, 10))

        control_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        control_frame.pack(fill="x", pady=(0, 10))

        buttons = [
            ("Start Monitoring", self.start_monitoring),
            ("View Bottlenecks", self.view_bottlenecks),
            ("Detect Anomalies", self.detect_anomalies),
            ("Start Tracking", self.start_tracking),
            ("Stop Tracking", self.stop_tracking),
            ("Export Data", self.export_data)
        ]

        for text, command in buttons:
            btn_color = "red" if text == "Stop Tracking" else None
            btn = ctk.CTkButton(
                control_frame,
                text=text,
                command=lambda cmd=command: self.handle_button_click(cmd),
                corner_radius=10,
                hover_color=("#0056b3", "#0056b3"),
                font=("Inter", 14, "bold"),
                width=200,
                height=50,
                fg_color=btn_color
            )
            btn.pack(pady=5)

        # Output frame
        self.output_frame = ctk.CTkFrame(self.control_panel, corner_radius=15, fg_color="#2c2c3e")
        self.output_frame.pack(fill="x", pady=10)

        self.output_text = ctk.CTkTextbox(
            self.output_frame,
            corner_radius=10,
            text_color="white",
            fg_color="#2c2c3e",
            font=("Cascadia Code", 14),
            scrollbar_button_color="#3c3c4e",
            height=300
        )
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.output_text.configure(state="disabled")
        self.output_text.bind("<Button-1>", self.swap_to_output_view)

        # Charts frame
        self.charts_frame = ctk.CTkFrame(self.content_frame)
        self.charts_frame.pack(side="right", fill="both", expand=True)

        self.create_performance_charts(self.charts_frame)
        self.is_swapped = False

    def handle_button_click(self, command):
        command()

    def swap_to_output_view(self, event):
        if not self.is_swapped:
            # Store current output content
            current_output = self.output_text.get("1.0", tk.END)

            # Remove current layout
            self.output_frame.pack_forget()
            self.charts_frame.pack_forget()

            # Create "Show Performance" label in original output space
            self.output_frame = ctk.CTkFrame(self.control_panel, corner_radius=15, fg_color="#2c2c3e")
            self.output_frame.pack(fill="x", pady=10)
            self.show_perf_label = ctk.CTkLabel(
                self.output_frame,
                text="Show Performance",
                font=("Cascadia Code", 14),
                text_color="white",
                height=300,
                width=190
            )
            self.show_perf_label.pack(fill="both", expand=True, padx=10, pady=10)
            self.show_perf_label.bind("<Button-1>", self.restore_normal_view)

            # Move output to charts space
            self.charts_frame = ctk.CTkFrame(self.content_frame, corner_radius=15, fg_color="#2c2c3e")
            self.charts_frame.pack(side="right", fill="both", expand=True)
            self.output_text = ctk.CTkTextbox(
                self.charts_frame,
                corner_radius=10,
                text_color="white",
                fg_color="#2c2c3e",
                font=("Cascadia Code", 14),
                scrollbar_button_color="#3c3c4e"
            )
            self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
            self.output_text.insert(tk.END, current_output)
            self.output_text.configure(state="disabled")
            self.output_text.bind("<Button-1>", self.swap_to_output_view)

            self.is_swapped = True

    def restore_normal_view(self, event):
        if self.is_swapped:
            # Store current output content
            current_output = self.output_text.get("1.0", tk.END)

            # Remove current layout
            self.output_frame.pack_forget()
            self.charts_frame.pack_forget()

            # Restore output in original space
            self.output_frame = ctk.CTkFrame(self.control_panel, corner_radius=15, fg_color="#2c2c3e")
            self.output_frame.pack(fill="x", pady=10)
            self.output_text = ctk.CTkTextbox(
                self.output_frame,
                corner_radius=10,
                text_color="white",
                fg_color="#2c2c3e",
                font=("Cascadia Code", 14),
                scrollbar_button_color="#3c3c4e",
                height=300
            )
            self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
            self.output_text.insert(tk.END, current_output)
            self.output_text.configure(state="disabled")
            self.output_text.bind("<Button-1>", self.swap_to_output_view)

            # Restore charts
            self.charts_frame = ctk.CTkFrame(self.content_frame)
            self.charts_frame.pack(side="right", fill="both", expand=True)
            self.create_performance_charts(self.charts_frame)

            self.is_swapped = False

    def create_performance_charts(self, parent):
        self.fig, axs = plt.subplots(2, 3, figsize=(10, 6), facecolor='#1f1f2b')
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.style.use('dark_background')

        metrics = [
            ('CPU Usage (%)', 'cpu_usage', (0, 0)),
            ('Memory Usage (%)', 'memory_usage', (0, 1)),
            ('Disk Read (MB/s)', 'disk_read', (0, 2)),
            ('Disk Write (MB/s)', 'disk_write', (1, 0)),
            ('Network Sent (MB/s)', 'network_sent', (1, 1)),
            ('Network Received (MB/s)', 'network_recv', (1, 2))
        ]

        self.performance_lines = {}
        for title, key, pos in metrics:
            ax = axs[pos[0], pos[1]]
            ax.set_title(title, color='white')
            ax.set_facecolor('#2c2c3e')
            ax.grid(True, linestyle='--', alpha=0.7)
            line, = ax.plot([], [], linewidth=2)
            self.performance_lines[key] = line

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def update_output(self, message):
        try:
            self.output_text.configure(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, message)
            self.output_text.configure(state="disabled")
        except Exception as e:
            print(f"Error updating output: {e}")

    def start_monitoring(self):
        self.update_output("Starting system monitoring...")

    def view_bottlenecks(self):
        self.update_output("Analyzing system bottlenecks...")

    def detect_anomalies(self):
        self.update_output("Detecting performance anomalies...\n[Expanded anomaly detection output here]")

    def start_tracking(self):
        if not self.is_tracking:
            self.is_tracking = True
            self.update_output("Real-time tracking started...")

    def stop_tracking(self):
        if self.is_tracking:
            self.is_tracking = False
            self.update_output("Real-time tracking stopped.")

    def export_data(self):
        self.update_output("Exporting performance data...")


if __name__ == "__main__":
    root = ctk.CTk()
    app = PerformanceAnalyzerApp(root)
    root.mainloop()