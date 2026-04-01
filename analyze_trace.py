import os
import sys
import glob
import sqlite3
import subprocess

def export_nsys_to_sqlite(nsys_rep_path):
    sqlite_path = nsys_rep_path.replace(".nsys-rep", ".sqlite")
    if os.path.exists(sqlite_path):
        return sqlite_path
    
    print(f"Exporting {os.path.basename(nsys_rep_path)} to SQLite...")
    cmd = f"nsys export --type sqlite --output {sqlite_path} {nsys_rep_path}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        return sqlite_path
    except subprocess.CalledProcessError as e:
        print(f"Export failed: {e}")
        return None

def merge_intervals(intervals):
    if not intervals: return 0
    intervals.sort()
    merged = []
    curr_start, curr_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return sum(end - start for start, end in merged)

def analyze_nsys_sqlite(sqlite_path, output_path):
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cursor.fetchall()}
    
    kernel_table = next((t for t in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"] if t in tables), None)
    nvtx_table = "NVTX_EVENTS" if "NVTX_EVENTS" in tables else None
    memcpy_table = "CUPTI_ACTIVITY_KIND_MEMCPY" if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else None

    valid_starts, valid_ends = [], []
    for table in filter(None, [kernel_table, nvtx_table]):
        cursor.execute(f"SELECT MIN(start), MAX(end) FROM {table}")
        start, end = cursor.fetchone()
        if start is not None: valid_starts.append(start); valid_ends.append(end)
    
    if not valid_starts:
        print("Error: No activity found in trace.")
        return

    wall_ms = (max(valid_ends) - min(valid_starts)) / 1e6
    
    gpu_active_ms = 0
    if kernel_table:
        cursor.execute(f"SELECT start, end FROM {kernel_table}")
        gpu_active_ms = merge_intervals(cursor.fetchall()) / 1e6
    
    idle_ms = wall_ms - gpu_active_ms
    gpu_util = (gpu_active_ms / wall_ms * 100) if wall_ms > 0 else 0

    report = [
        "="*80, "VLM Profiler Optimization Report", "="*80,
        f"Duration: {wall_ms:>10.2f} ms",
        f"GPU Active: {gpu_active_ms:>10.2f} ms ({gpu_util:.1f}%)",
        f"GPU Idle: {idle_ms:>10.2f} ms ({100-gpu_util:.1f}%)\n"
    ]

    if nvtx_table:
        report.append("Top 10 NVTX Ranges:")
        cursor.execute(f"""
            SELECT COALESCE(text, StringIds.value, 'Unknown'), SUM(end - start)/1e6 as total_ms, COUNT(*) 
            FROM {nvtx_table} LEFT JOIN StringIds ON {nvtx_table}.textId = StringIds.id
            GROUP BY 1 ORDER BY total_ms DESC LIMIT 10
        """)
        for name, total, count in cursor.fetchall():
            report.append(f"{str(name)[:30]:<30} | {total:>10.2f} ms | {count:>5} calls")
        report.append("")

    if kernel_table:
        report.append("Top 10 CUDA Kernels:")
        cursor.execute(f"""
            SELECT StringIds.value, SUM(end - start)/1e6 as total_ms, COUNT(*) 
            FROM {kernel_table} JOIN StringIds ON {kernel_table}.shortName = StringIds.id
            GROUP BY 1 ORDER BY total_ms DESC LIMIT 10
        """)
        for name, total, count in cursor.fetchall():
            report.append(f"{str(name)[:60]:<60} | {total:>10.2f} ms | {count:>5}")
        report.append("")

    with open(output_path, "w") as f: f.write(full_report)
    conn.close()

if __name__ == "__main__":
    BASE_DIR = "/data/mihejkoveg/VLMProfiler_runs"
    reps = glob.glob(os.path.join(BASE_DIR, "*.nsys-rep"))
    if not reps:
        print("No .nsys-rep files found."); sys.exit(1)
    
    latest_rep = max(reps, key=os.path.getmtime)
    sqlite_file = export_nsys_to_sqlite(latest_rep)
    if sqlite_file:
        analyze_nsys_sqlite(sqlite_file, os.path.join(BASE_DIR, "optimization_report.txt"))
