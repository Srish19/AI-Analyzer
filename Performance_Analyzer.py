import tkinter as tk
from tkinter import filedialog
import psutil
import time
from datetime import datetime
import pandas as pd
import customtkinter as ctk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class EnhancedGUI:
    def __init__(self, root):
        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("AI-Powered Performance Analyzer")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1f1f2b")  # Dark background

        # Initialize data attribute
        self.data = None

        # Header with gradient effect
        header_frame = ctk.CTkFrame(root,
                                    fg_color="transparent",
                                    bg_color="transparent"
                                    )
        header_frame.pack(fill="x", pady=(0, 10))

        header = ctk.CTkLabel(
            header_frame,
            text="AI-Powered Performance Analyzer",
            font=("Inter", 24, "bold"),
            text_color="white",
            bg_color="transparent"
        )
        header.pack(pady=15, fill="x")

        # Buttons Frame with modern styling
        button_frame = ctk.CTkFrame(
            root,
            fg_color="transparent",
            bg_color="transparent"
        )
        button_frame.pack(pady=10)

        buttons = [
            ("Start Monitoring", self.start_monitoring),
            ("View Bottlenecks", self.view_bottlenecks),
            ("Detect Anomalies", self.detect_anomalies),
            ("Export Data", self.export_data)
        ]

        for text, command in buttons:
            btn = ctk.CTkButton(
                button_frame,
                text=text,
                command=command,
                corner_radius=10,
                hover_color=("#0056b3", "#0056b3"),
                font=("Inter", 14, "bold"),
                width=200,
                height=50
            )
            btn.pack(side="left", padx=10)

        # Output Section with improved styling
        output_frame = ctk.CTkFrame(
            root,
            corner_radius=15,
            fg_color="#2c2c3e",
            bg_color="#1f1f2b"
        )
        output_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Scrollable text widget
        self.output_text = ctk.CTkTextbox(
            output_frame,
            corner_radius=10,
            text_color="white",
            fg_color="#2c2c3e",
            font=("Cascadia Code", 14),
            scrollbar_button_color="#3c3c4e"
        )
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Initial disabled state
        self.output_text.configure(state="disabled")

    def update_output(self, message):
        """
        Update the output text widget with a given message.

        Args:
            message (str): Message to display in the output text widget
        """
        try:
            self.output_text.configure(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, message)
            self.output_text.configure(state="disabled")
        except Exception as e:
            print(f"Error updating output: {e}")

    def start_monitoring(self):
        """
        Start system performance monitoring for 20 seconds.
        Collects and displays system metrics.
        """
        try:
            self.update_output("Starting to monitor system metrics for 20 seconds...")
            self.data = self.monitor_for_20_seconds()

            if self.data is not None and not self.data.empty:
                self.update_output("Data collection completed!")
                self.update_output("\nCollected Data (first 5 rows):")
                self.update_output(self.data.head().to_string(index=False))
            else:
                self.update_output("No data was collected during monitoring.")
        except Exception as e:
            self.update_output(f"Monitoring error: {str(e)}")

    def view_bottlenecks(self):
        """
        Identify and display system bottlenecks based on collected data.
        """
        if self.data is None:
            self.update_output("No data available. Please start monitoring first.")
            return

        try:
            bottlenecks = self.identify_bottlenecks(self.data)
            if bottlenecks.empty:
                self.update_output("No bottlenecks detected.")
            else:
                self.update_output("Bottlenecks Detected:\n")
                self.update_output(bottlenecks.to_string(index=False))
        except Exception as e:
            self.update_output(f"Bottleneck analysis error: {str(e)}")

    def detect_anomalies(self):
        """
        Detect anomalies in system performance using machine learning.
        """
        if self.data is None:
            self.update_output("No data available. Please start monitoring first.")
            return

        try:
            # Label data: Assume CPU > 75% or Memory > 75% is an anomaly
            self.data["Anomaly"] = ((self.data["CPU_Usage"] > 75) | (self.data["Memory_Percent"] > 75)).astype(int)

            # Prepare features and labels
            X = self.data.drop(columns=["Timestamp", "Anomaly"])
            y = self.data["Anomaly"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Display results
            self.update_output(f"Random Forest Model Trained. Accuracy: {accuracy:.2f}")

            # Predict anomalies in collected data
            X_scaled = scaler.transform(X)
            self.data["Predicted_Anomaly"] = model.predict(X_scaled)

            anomalies = self.data[self.data["Predicted_Anomaly"] == 1]
            if anomalies.empty:
                self.update_output("No anomalies detected.")
            else:
                self.update_output("Detected Anomalies:\n")
                self.update_output(anomalies.to_string(index=False))
        except Exception as e:
            self.update_output(f"Anomaly detection error: {str(e)}")

    def export_data(self):
        """
        Export collected system performance data to a CSV file.
        """
        if self.data is None:
            self.update_output("No data available to export. Please start monitoring first.")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.data.to_csv(file_path, index=False)
                self.update_output(f"Data successfully exported to {file_path}")
        except Exception as e:
            self.update_output(f"Data export error: {str(e)}")

    @staticmethod
    def monitor_for_20_seconds():
        """
        Collect system performance metrics for 20 seconds.

        Returns:
            pd.DataFrame: Collected system performance data
        """
        start_time = time.time()
        data = []
        try:
            while (time.time() - start_time) < 20:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
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
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Monitoring error: {e}")
            return pd.DataFrame()

    @staticmethod
    def identify_bottlenecks(data):
        """
        Identify system performance bottlenecks.

        Args:
            data (pd.DataFrame): System performance data

        Returns:
            pd.DataFrame: Rows representing bottlenecks
        """
        return data[(data['CPU_Usage'] > 75) | (data['Memory_Percent'] > 75)]



def main():
    """
    Main application entry point.
    """
    root = ctk.CTk()
    app = EnhancedGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()