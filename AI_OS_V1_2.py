import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import psutil
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import threading
import math
import random

class PerformanceAnalyzerApp:
    def __init__(self, root):
        self.is_tracking = False
        self.charts_frame = None
        self.data = None
        self.root = root
        self.root.title("AI-Powered System Performance Analyzer")
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
        title_label.place(relx=0.5, rely=0.40, anchor="center")

        welcome_label = ctk.CTkLabel(
            self.welcome_frame,
            text="The Future Is Now.",
            font=("Inter", 24),
            text_color="white"
        )
        welcome_label.place(relx=0.5, rely=0.45, anchor="center")

        self.enter_button = ctk.CTkButton(
            self.welcome_frame,
            text="ENTER",
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
        self.root.attributes('-fullscreen', True)
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
            ("AI Performance", self.detect_anomalies),
            ("Start Live Tracking", self.start_tracking),
            ("Stop Tracking", self.stop_tracking),
            ("Export Data", self.export_data),
            ("Exit", self.exit_program)
        ]

        for text, command in buttons:
            btn_color = "red" if text in ["Stop Tracking", "Exit"] else None
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

        self.charts_frame = ctk.CTkFrame(self.content_frame, corner_radius=15, fg_color="#2c2c3e")
        self.charts_frame.pack(side="right", fill="both", expand=True)
        self.create_performance_charts(self.charts_frame)

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
        self.output_text.insert(tk.END, "Welcome to the AI Performance Analyzer.\n\nClick here to view output.")
        self.output_text.configure(state="disabled")
        self.output_text.bind("<Button-1>", self.swap_to_output_view)

        self.is_swapped = False

    def handle_button_click(self, command):
        command()

    def swap_to_output_view(self, event):
        if not self.is_swapped:
            current_output = self.output_text.get("1.0", tk.END)
            self.output_frame.pack_forget()
            self.charts_frame.pack_forget()

            self.charts_frame = ctk.CTkFrame(self.control_panel, corner_radius=15, fg_color="#2c2c3e")
            self.charts_frame.pack(fill="x", pady=10)
            self.show_charts_label = ctk.CTkLabel(
                self.charts_frame,
                text="Show Performance Charts",
                font=("Cascadia Code", 14),
                text_color="white",
                height=300,
                width=190
            )
            self.show_charts_label.pack(fill="both", expand=True, padx=10, pady=10)
            self.show_charts_label.bind("<Button-1>", self.swap_to_charts_view)

            self.output_frame = ctk.CTkFrame(self.content_frame, corner_radius=15, fg_color="#2c2c3e")
            self.output_frame.pack(side="right", fill="both", expand=True)
            self.output_text = ctk.CTkTextbox(
                self.output_frame,
                corner_radius=10,
                text_color="white",
                fg_color="#2c2c3e",
                font=("Cascadia Code", 14),
                scrollbar_button_color="#3c3c4e"
            )
            self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
            self.output_text.insert(tk.END, current_output)
            self.output_text.configure(state="disabled")
            self.output_text.bind("<Button-1>", self.swap_to_charts_view)

            self.is_swapped = True

    def swap_to_charts_view(self, event):
        if self.is_swapped:
            current_output = self.output_text.get("1.0", tk.END)
            self.output_frame.pack_forget()
            self.charts_frame.pack_forget()

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

            self.charts_frame = ctk.CTkFrame(self.content_frame, corner_radius=15, fg_color="#2c2c3e")
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
        self.update_output("Starting system monitoring for 20 seconds...")
        self.progress_bar = ctk.CTkProgressBar(self.control_panel, width=200)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        self.start_tracking()

        monitor_thread = threading.Thread(target=self.monitor_for_20_seconds, daemon=True)
        monitor_thread.start()

        start_time = time.time()
        duration = 20

        def update_progress():
            elapsed = time.time() - start_time
            if elapsed < duration:
                progress = elapsed / duration
                self.progress_bar.set(progress)
                self.root.after(100, update_progress)
            else:
                self.progress_bar.set(1)
                self.root.after(100, lambda: self.progress_bar.pack_forget())
                self.stop_tracking()

        self.root.after(100, update_progress)

    def monitor_for_20_seconds(self):
        start_time = time.time()
        data = []
        try:
            while (time.time() - start_time) < 20:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cpu_usage = psutil.cpu_percent(interval=1) + random.uniform(-5, 5)  # Add slight noise
                cpu_usage = max(0, min(100, cpu_usage))  # Clamp to 0-100
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent + random.uniform(-5, 5)  # Add slight noise
                memory_percent = max(0, min(100, memory_percent))  # Clamp to 0-100
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                data.append({
                    "Timestamp": timestamp,
                    "CPU_Usage": cpu_usage,
                    "Memory_Percent": memory_percent,
                    "Disk_Read_MB": disk_io.read_bytes / 1024 / 1024,
                    "Disk_Write_MB": disk_io.write_bytes / 1024 / 1024,
                    "Network_Sent_MB": net_io.bytes_sent / 1024 / 1024,
                    "Network_Received_MB": net_io.bytes_recv / 1024 / 1024
                })
            self.data = pd.DataFrame(data)
            if not self.data.empty:
                self.update_output("Data collection completed!\n\nCollected Data (first 5 rows):\n" +
                                  self.data.head().to_string(index=False))
            else:
                self.update_output("No data was collected during monitoring.")
        except Exception as e:
            self.update_output(f"Monitoring error: {str(e)}")
            self.data = pd.DataFrame()

    @staticmethod
    def identify_bottlenecks(data):
        return data[(data['CPU_Usage'] > 75) | (data['Memory_Percent'] > 75)]

    def view_bottlenecks(self):
        if self.data is None:
            self.update_output("No data available. Please start monitoring first.")
            return

        try:
            bottlenecks = self.identify_bottlenecks(self.data)
            if bottlenecks.empty:
                self.update_output("No bottlenecks detected.")
                if hasattr(self, 'canvas'):
                    self.fig.clf()
                    self.canvas.draw()
            else:
                self.update_output("Bottlenecks Detected:\n" + bottlenecks.to_string(index=False))

                if hasattr(self, 'canvas'):
                    self.fig.clf()
                else:
                    self.fig = plt.Figure(figsize=(10, 6), facecolor='#1f1f2b')

                ax1 = self.fig.add_subplot(211)
                ax1.plot(self.data['Timestamp'], self.data['CPU_Usage'], label='CPU Usage (%)', color='cyan')
                ax1.scatter(bottlenecks['Timestamp'], bottlenecks['CPU_Usage'], color='red', label='Bottlenecks',
                            zorder=5)
                ax1.set_title('CPU Usage Over Time', color='white')
                ax1.set_ylabel('CPU Usage (%)', color='white')
                ax1.tick_params(axis='x', rotation=45, labelcolor='white')
                ax1.tick_params(axis='y', labelcolor='white')
                ax1.set_facecolor('#2c2c3e')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()

                ax2 = self.fig.add_subplot(212)
                ax2.plot(self.data['Timestamp'], self.data['Memory_Percent'], label='Memory Usage (%)', color='yellow')
                ax2.scatter(bottlenecks['Timestamp'], bottlenecks['Memory_Percent'], color='red', label='Bottlenecks',
                            zorder=5)
                ax2.set_title('Memory Usage Over Time', color='white')
                ax2.set_xlabel('Timestamp', color='white')
                ax2.set_ylabel('Memory Usage (%)', color='white')
                ax2.tick_params(axis='x', rotation=45, labelcolor='white')
                ax2.tick_params(axis='y', labelcolor='white')
                ax2.set_facecolor('#2c2c3e')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()

                self.fig.tight_layout()

                if hasattr(self, 'canvas'):
                    self.canvas.draw()
                else:
                    if self.is_swapped:
                        self.swap_to_charts_view(None)
                    self.canvas = FigureCanvasTkAgg(self.fig, master=self.charts_frame)
                    self.canvas_widget = self.canvas.get_tk_widget()
                    self.canvas_widget.pack(fill="both", expand=True)
                    self.canvas.draw()

        except Exception as e:
            self.update_output(f"Bottleneck analysis error: {str(e)}")

    def create_performance_gauge(self, score):
        if hasattr(self, 'canvas'):
            self.fig.clf()
        else:
            self.fig = plt.Figure(figsize=(6, 6), facecolor='#1f1f2b')

        ax = self.fig.add_subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks([])

        ax.set_xticks(np.linspace(0, 2 * np.pi, 11))
        ax.set_xticklabels([f"{i}%" for i in range(0, 101, 10)], color='white')

        ax.bar(np.linspace(0, 2 * np.pi, 100), np.ones(100) * 100, width=0.1, color='gray', alpha=0.3)
        score_angle = (score / 100) * 2 * np.pi
        ax.bar(np.linspace(0, score_angle, 100), np.ones(100) * 100, width=0.1, color='cyan', alpha=0.8)
        ax.plot([score_angle, score_angle], [0, 80], color='red', lw=2)

        ax.set_title(f"Performance Score: {score:.1f}%", color='white', pad=20)
        ax.set_facecolor('#2c2c3e')

        def update_charts():
            if self.is_swapped:
                self.swap_to_charts_view(None)
            if hasattr(self, 'canvas'):
                self.canvas.draw()
            else:
                self.canvas = FigureCanvasTkAgg(self.fig, master=self.charts_frame)
                self.canvas_widget = self.canvas.get_tk_widget()
                self.canvas_widget.pack(fill="both", expand=True)
                self.canvas.draw()

        self.root.after(0, update_charts)

    def detect_anomalies(self):
        if self.data is None:
            self.update_output("No data available. Please start monitoring first.")
            return

        try:
            # Set anomaly labels with controlled randomness
            random.seed(42)  # Fixed seed for reproducibility
            self.data["Anomaly"] = ((self.data["CPU_Usage"] > 75) | (self.data["Memory_Percent"] > 75)).astype(int)
            # Flip 10% of labels randomly for slight noise
            for i in range(len(self.data)):
                if random.random() < 0.1:  # Reduced from 0.2 to 0.1
                    self.data.loc[i, "Anomaly"] = 1 - self.data.loc[i, "Anomaly"]

            X = self.data.drop(columns=["Timestamp", "Anomaly"])
            y = self.data["Anomaly"]

            # Use cross-validation for more stable accuracy
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)  # Adjusted parameters
            cv_scores = cross_val_score(model, X_scaled, y, cv=5)  # 5-fold cross-validation
            accuracy = np.mean(cv_scores)

            # Train final model for predictions
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            self.data["Predicted_Anomaly"] = model.predict(X_scaled)

            cpu_avg = self.data["CPU_Usage"].mean()
            mem_avg = self.data["Memory_Percent"].mean()
            disk_read_avg = self.data["Disk_Read_MB"].mean()
            disk_write_avg = self.data["Disk_Write_MB"].mean()
            net_sent_avg = self.data["Network_Sent_MB"].mean()
            net_recv_avg = self.data["Network_Received_MB"].mean()

            cpu_score = max(0, 100 - cpu_avg)
            mem_score = max(0, 100 - mem_avg)
            disk_score = min(100, (disk_read_avg + disk_write_avg) / 10)
            net_score = min(100, (net_sent_avg + net_recv_avg) / 10)

            performance_score = (0.4 * cpu_score + 0.4 * mem_score + 0.1 * disk_score + 0.1 * net_score)

            output_message = (
                f"Performance Analysis:\n"
                f"Random Forest Model Accuracy (Cross-Validated): {accuracy:.2f}\n"
                f"Avg CPU Usage: {cpu_avg:.1f}%\n"
                f"Avg Memory Usage: {mem_avg:.1f}%\n"
                f"Avg Disk I/O: {disk_read_avg + disk_write_avg:.2f} MB/s\n"
                f"Avg Network I/O: {net_sent_avg + net_recv_avg:.2f} MB/s\n"
                f"Performance Score: {performance_score:.1f}% (Higher is better)\n"
            )
            anomalies = self.data[self.data["Predicted_Anomaly"] == 1]
            if anomalies.empty:
                output_message += "No anomalies detected."
            else:
                output_message += f"Detected Anomalies ({len(anomalies)} points):\n" + anomalies.to_string(index=False)
            self.update_output(output_message)

            self.create_performance_gauge(performance_score)

        except Exception as e:
            self.update_output(f"Anomaly detection error: {str(e)}")

    def start_tracking(self):
        if not self.is_tracking:
            self.is_tracking = True
            self.performance_data = {
                'timestamps': [], 'cpu_usage': [], 'memory_usage': [],
                'disk_read': [], 'disk_write': [], 'network_sent': [], 'network_recv': []
            }
            if not hasattr(self, 'performance_lines'):
                self.create_performance_charts(self.charts_frame)
            for line in self.performance_lines.values():
                line.set_data([], [])

            self.tracking_thread = threading.Thread(target=self.collect_performance_data, daemon=True)
            self.tracking_thread.start()
            self.update_output("Real-time tracking started...")

    def stop_tracking(self):
        if self.is_tracking:
            self.is_tracking = False
            if self.tracking_thread:
                self.tracking_thread.join()
            self.update_output("Real-time tracking stopped.")

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

                self.root.after(0, self.update_charts)
                time.sleep(1)
            except Exception as e:
                print(f"Error collecting performance data: {e}")
                break

    def update_charts(self):
        metrics = ['cpu_usage', 'memory_usage', 'disk_read', 'disk_write', 'network_sent', 'network_recv']
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
        if self.data is None and not self.performance_data['timestamps']:
            self.update_output("No data available to export.")
            return
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                if self.data is not None:
                    self.data.to_csv(file_path, index=False)
                else:
                    df = pd.DataFrame(self.performance_data)
                    df.to_csv(file_path, index=False)
                self.update_output(f"Data exported successfully to {file_path}")
        except Exception as e:
            self.update_output(f"Error exporting data: {str(e)}")

    def exit_program(self):
        if messagebox.askyesno("Exit Confirmation", "Are you sure you want to exit the program?"):
            if self.is_tracking:
                self.stop_tracking()
            self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = PerformanceAnalyzerApp(root)
    root.mainloop()
