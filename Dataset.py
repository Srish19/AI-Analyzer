import psutil
import time
from datetime import datetime

def format_bytes(size):
    """Convert bytes to a readable format (KB, MB, GB, etc.)"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

def monitor_for_20_seconds():
    print("Collecting system metrics for 20 seconds...")
    print("-" * 60)
    start_time = time.time()  # Record the start time

    data = []  # To store collected metrics

    while (time.time() - start_time) < 20:  # Run for 20 seconds
        # Current Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # CPU Utilization
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_cores = psutil.cpu_count(logical=True)

        # Memory Usage
        memory_info = psutil.virtual_memory()
        total_memory = format_bytes(memory_info.total)
        used_memory = format_bytes(memory_info.used)
        memory_percent = memory_info.percent

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        read_bytes = format_bytes(disk_io.read_bytes)
        write_bytes = format_bytes(disk_io.write_bytes)

        # Network Usage
        net_io = psutil.net_io_counters()
        sent_bytes = format_bytes(net_io.bytes_sent)
        recv_bytes = format_bytes(net_io.bytes_recv)

        # Store the collected data
        data.append({
            "Timestamp": timestamp,
            "CPU_Usage": cpu_usage,
            "CPU_Cores": cpu_cores,
            "Total_Memory": total_memory,
            "Used_Memory": used_memory,
            "Memory_Percent": memory_percent,
            "Disk_Read_Bytes": read_bytes,
            "Disk_Write_Bytes": write_bytes,
            "Network_Sent": sent_bytes,
            "Network_Received": recv_bytes
        })

        # Print the metrics
        print(f"[{timestamp}] CPU: {cpu_usage}%, Memory: {memory_percent}% used, Disk I/O: Read={read_bytes}, Write={write_bytes}, Network: Sent={sent_bytes}, Received={recv_bytes}")

    print("-" * 60)
    print("Data collection complete!")
    return data

# Run the function
collected_data = monitor_for_20_seconds()

# Optional: Save to a CSV file
import pandas as pd
df = pd.DataFrame(collected_data)
df.to_csv("system_metrics_20_seconds.csv", index=False)
print("Metrics saved to 'system_metrics_20_seconds.csv'")

