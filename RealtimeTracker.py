import tkinter as tk
import customtkinter as ctk
import psutil
import threading
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class RealTimePerformanceTracker:
    def __init__(self, root):
        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("Real-Time System Performance Tracker")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1f1f2b")

        # Tracking variables
        self.is_tracking = False
        self.tracking_thread = None
        self.performance_data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'disk_read': [],
            'disk_write': [],
            'network_sent': [],
            'network_recv': []
        }

        # Create main layout
        self.create_ui()

    def create_ui(self):
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        control_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        control_frame.pack(fill="x", pady=10)

        start_btn = ctk.CTkButton(control_frame, text="Start Tracking", command=self.start_tracking, corner_radius=10)
        start_btn.pack(side="left", padx=10)

        stop_btn = ctk.CTkButton(control_frame, text="Stop Tracking", command=self.stop_tracking, corner_radius=10, fg_color="red")
        stop_btn.pack(side="left", padx=10)

        export_btn = ctk.CTkButton(control_frame, text="Export Data", command=self.export_data, corner_radius=10)
        export_btn.pack(side="left", padx=10)

        metrics_frame = ctk.CTkFrame(main_frame)
        metrics_frame.pack(fill="both", expand=True, pady=10)

        self.create_performance_charts(metrics_frame)

    def create_performance_charts(self, parent):
        self.fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor='#1f1f2b')
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

    def start_tracking(self):
        if not self.is_tracking:
            self.is_tracking = True
            self.performance_data = {
                'timestamps': [],
                'cpu_usage': [],
                'memory_usage': [],
                'disk_read': [],
                'disk_write': [],
                'network_sent': [],
                'network_recv': []
            }

            for line in self.performance_lines.values():
                line.set_data([], [])

            self.tracking_thread = threading.Thread(target=self.collect_performance_data)
            self.tracking_thread.start()

    def stop_tracking(self):
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join()

    def collect_performance_data(self):
        disk_counters_prev = psutil.disk_io_counters()
        net_counters_prev = psutil.net_io_counters()

        while self.is_tracking:
            try:
                current_time = time.time()
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent

                disk_counters = psutil.disk_io_counters()
                disk_read = (disk_counters.read_bytes - disk_counters_prev.read_bytes) / (1024 * 1024)
                disk_write = (disk_counters.write_bytes - disk_counters_prev.write_bytes) / (1024 * 1024)
                disk_counters_prev = disk_counters

                net_counters = psutil.net_io_counters()
                network_sent = (net_counters.bytes_sent - net_counters_prev.bytes_sent) / (1024 * 1024)
                network_recv = (net_counters.bytes_recv - net_counters_prev.bytes_recv) / (1024 * 1024)
                net_counters_prev = net_counters

                self.performance_data['timestamps'].append(current_time)
                self.performance_data['cpu_usage'].append(cpu_usage)
                self.performance_data['memory_usage'].append(memory_usage)
                self.performance_data['disk_read'].append(disk_read)
                self.performance_data['disk_write'].append(disk_write)
                self.performance_data['network_sent'].append(network_sent)
                self.performance_data['network_recv'].append(network_recv)

                self.update_charts()
                time.sleep(1)

            except Exception as e:
                print(f"Error collecting performance data: {e}")
                break

    def update_charts(self):
        metrics = [
            'cpu_usage', 'memory_usage', 'disk_read',
            'disk_write', 'network_sent', 'network_recv'
        ]

        for metric in metrics:
            line = self.performance_lines[metric]
            data = self.performance_data[metric][-60:]
            timestamps = self.performance_data['timestamps'][-60:]

            if data:
                line.set_data(timestamps, data)
                ax = line.axes
                ax.relim()
                ax.autoscale_view()

        self.canvas.draw()
        self.canvas.flush_events()

    def export_data(self):
        if not self.performance_data['timestamps']:
            print("No data to export")
            return

        df = pd.DataFrame(self.performance_data)
        df['timestamp'] = pd.to_datetime(df['timestamps'], unit='s')

        export_path = f"system_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(export_path, index=False)
        print(f"Data exported to {export_path}")

def main():
    root = ctk.CTk()
    app = RealTimePerformanceTracker(root)
    root.mainloop()

if __name__ == "__main__":
    main()
