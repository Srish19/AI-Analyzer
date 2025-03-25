import psutil
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

def monitor_for_20_seconds():
    print("Collecting system metrics for 20 seconds...")
    print("-" * 60)

    start_time = time.time()
    data = []

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

        print(f"[{timestamp}] CPU: {cpu_usage}%, Memory: {memory_percent}%, Disk: Read={disk_io.read_bytes/1024/1024:.2f}MB, Write={disk_io.write_bytes/1024/1024:.2f}MB, Network: Sent={net_io.bytes_sent/1024/1024:.2f}MB, Received={net_io.bytes_recv/1024/1024:.2f}MB")

    print("-" * 60)
    print("Data collection complete!")
    return pd.DataFrame(data)

def identify_bottlenecks(data):
    return data[(data['CPU_Usage'] > 75) | (data['Memory_Percent'] > 75)]

def forecast_resources(data, target_col):
    timestamps = np.arange(len(data)).reshape(-1, 1)
    values = data[target_col].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_timestamps = scaler.fit_transform(timestamps)
    scaled_values = scaler.fit_transform(values)

    model = LinearRegression()
    model.fit(scaled_timestamps, scaled_values)
    predictions = model.predict(scaled_timestamps)

    return predictions * scaler.scale_[0] + scaler.mean_

def optimize_disk_and_network(data):
    optimizations = []
    for index, row in data.iterrows():
        if row['Disk_Read_MB'] > 50:
            optimizations.append(f"High Disk Read at {row['Timestamp']}: Reduce read operations or optimize disk access.")
        if row['Disk_Write_MB'] > 50:
            optimizations.append(f"High Disk Write at {row['Timestamp']}: Consider defragmentation or limiting write operations.")
        if row['Network_Sent_MB'] > 10:
            optimizations.append(f"High Network Upload at {row['Timestamp']}: Optimize upload tasks or limit bandwidth usage.")
        if row['Network_Received_MB'] > 10:
            optimizations.append(f"High Network Download at {row['Timestamp']}: Limit download bandwidth or schedule tasks during off-peak hours.")
    return optimizations

def detect_anomalies(data):
    anomalies = []
    for col in ['CPU_Usage', 'Memory_Percent', 'Disk_Read_MB', 'Disk_Write_MB', 'Network_Sent_MB', 'Network_Received_MB']:
        mean = data[col].mean()
        std = data[col].std()
        z_scores = (data[col] - mean) / std

        for i, z in enumerate(z_scores):
            if abs(z) > 2:
                anomalies.append(f"Anomaly detected in {col} at {data.iloc[i]['Timestamp']}: {data.iloc[i][col]:.2f}")
    return anomalies

def visualize_data(data, predictions=None):
    sns.set(style="whitegrid", context="notebook")
    fig, axs = plt.subplots(4, 1, figsize=(14, 12))

    axs[0].plot(data.index, data['CPU_Usage'], label='CPU Usage (%)', color='#007acc', linewidth=2.5)
    axs[0].plot(data.index, data['Memory_Percent'], label='Memory Usage (%)', color='#ff5733', linewidth=2.5)
    axs[0].set_title("CPU and Memory Usage", fontsize=14, weight='bold', color="#333333")
    axs[0].set_ylabel("Usage (%)", fontsize=12, color="#555555")
    axs[0].legend(loc='upper left', fontsize=10, frameon=False)
    axs[0].grid(True, linestyle='--', alpha=0.7)

    axs[1].plot(data.index, data['Disk_Read_MB'], label='Disk Read (MB)', color='#28a745', linewidth=2.5)
    axs[1].plot(data.index, data['Disk_Write_MB'], label='Disk Write (MB)', color='#dc3545', linewidth=2.5)
    axs[1].set_title("Disk I/O", fontsize=14, weight='bold', color="#333333")
    axs[1].set_ylabel("Activity (MB)", fontsize=12, color="#555555")
    axs[1].legend(loc='upper left', fontsize=10, frameon=False)
    axs[1].grid(True, linestyle='--', alpha=0.7)

    axs[2].plot(data.index, data['Network_Sent_MB'], label='Network Sent (MB)', color='#6f42c1', linewidth=2.5)
    axs[2].plot(data.index, data['Network_Received_MB'], label='Network Received (MB)', color='#e83e8c', linewidth=2.5)
    axs[2].set_title("Network I/O", fontsize=14, weight='bold', color="#333333")
    axs[2].set_ylabel("Activity (MB)", fontsize=12, color="#555555")
    axs[2].legend(loc='upper left', fontsize=10, frameon=False)
    axs[2].grid(True, linestyle='--', alpha=0.7)

    axs[3].scatter(data['CPU_Usage'], data['Memory_Percent'], color='Maroon', alpha=0.4)
    axs[3].set_title("Correlation Between CPU Usage and Memory Usage", fontsize=14, weight='bold', color="#333333")
    axs[3].set_xlabel("CPU Usage (%)", fontsize=12, color="#555555")
    axs[3].set_ylabel("Memory Usage (%)", fontsize=12, color="#555555")
    axs[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(pad=3.0)
    plt.show()
    fig.savefig("system_metrics_visualization_with_scatter.png", dpi=300)
    print("Visualization saved as 'system_metrics_visualization_with_scatter.png'")

def main():
    collected_data = monitor_for_20_seconds()

    bottlenecks = identify_bottlenecks(collected_data)
    if not bottlenecks.empty:
        print("\nDetected Bottlenecks:")
        print(bottlenecks)
    else:
        print("\nNo bottlenecks detected.")

    anomalies = detect_anomalies(collected_data)
    if anomalies:
        print("\nAnomalies Detected:")
        for anomaly in anomalies:
            print(anomaly)
    else:
        print("\nNo anomalies detected.")

    disk_network_optimizations = optimize_disk_and_network(collected_data)
    if disk_network_optimizations:
        print("\nDisk and Network Optimization Suggestions:")
        for optimization in disk_network_optimizations:
            print(optimization)
    else:
        print("\nNo Disk or Network optimizations required.")

    memory_predictions = forecast_resources(collected_data, 'Memory_Percent')

    print("\nVisualizing Data...")
    visualize_data(collected_data, predictions=memory_predictions)

if __name__ == "__main__":
    main()
