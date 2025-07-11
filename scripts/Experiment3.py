import subprocess
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. User Settings ---

NAME = "Cheddar"

# (Important) Enter the actual path to the workload executables.
BUILD_DIR = "./unittest/build"

# List of workloads to test
workloads = {
    "Bts": "./boot_test",
    "HELR": "./helr",
    "ResNet": "./resnet",
    "Sort": "./sorting",
}

# log_delta values tested for each workload execution (in order of output)
LOG_DELTAS = [30, 35, 40, 48]

# (Important) Library versions to test and their absolute file paths
BASE_VERSION_NAME = "Base"  # Name of the baseline version
lib_versions = {
    BASE_VERSION_NAME: "/cheddar/lib/base/libcheddar_base.so",
    "+SeqFuse": "/cheddar/lib/seqfuse/libcheddar_seqfuse.so",
    "+ParFuse": "/cheddar/lib/parfuse/libcheddar_parfuse.so",
}

# The name of the symbolic link to be changed
SYMLINK_NAME = "/cheddar/lib/libcheddar.so"


# --- 2. Function Definitions ---


def run_and_parse_workload(executable_path):
    """Runs the workload, parses all 'Wall clock time' from the output, and returns them as a list."""
    command = [executable_path]
    execution_times = []
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, cwd=BUILD_DIR
        )
        output = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  -> Error running {os.path.basename(executable_path)}: {e}")
        output = ""
        if hasattr(e, "stdout") and e.stdout:
            output += e.stdout
        if hasattr(e, "stderr") and e.stderr:
            output += e.stderr

    for line in output.splitlines():
        if "Wall clock time" in line:
            time_part = line.split()[-1]
            execution_times.append(float(time_part.replace("us", "")))

    return execution_times


def symlink_force(target, link_name):
    """Forcibly overwrites and creates a new symbolic link."""
    try:
        if os.path.lexists(link_name):
            os.remove(link_name)
        os.symlink(target, link_name)
    except Exception as e:
        print(f"Fatal: Failed to create symlink {link_name} -> {target}. Error: {e}")
        raise


def normalize_and_plot(df, gpu_name):
    """Normalizes the data and generates grouped bar plots for each log_delta value."""
    if df.empty:
        print("No data to plot.")
        return

    # 1. Normalize based on the 'Base' version's time
    # Find the Base time for each group based on 'Workload' and 'log_delta'
    base_times = df[df["Version"] == BASE_VERSION_NAME].set_index(
        ["Workload", "log_delta"]
    )["Time (us)"]
    df_indexed = df.set_index(["Workload", "log_delta"])
    df_indexed["Base Time"] = df_indexed.index.map(base_times)

    # Prevent division by zero
    df_indexed["Base Time"].replace(0, float("nan"), inplace=True)

    df_indexed["Relative execution time"] = (
        df_indexed["Time (us)"] / df_indexed["Base Time"]
    )
    df_normalized = df_indexed.reset_index()

    # 2. Generate a separate plot for each log_delta value
    for log_delta in df_normalized["log_delta"].unique():
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")

        subset_df = df_normalized[df_normalized["log_delta"] == log_delta]

        ax = sns.barplot(
            data=subset_df,
            x="Workload",
            y="Relative execution time",
            hue="Version",
            hue_order=lib_versions.keys(),  # Fix the legend order
        )

        for patch in ax.patches:
            current_width = patch.get_width()
            current_x = patch.get_x()

            # Set the desired new width (can be adjusted)
            new_width = 0.25

            # Apply the new width
            patch.set_width(new_width)

            # Adjust the x position to center the bar
            patch.set_x(current_x + (current_width - new_width) / 2)

        # Graph design (reflecting requirements)
        ax.set_title(
            f"Performance on {gpu_name} (log_delta = {log_delta})", fontsize=16
        )
        ax.set_ylabel("Relative execution time", fontsize=12)
        ax.set_xlabel("Workload", fontsize=12)
        ax.axhline(
            1.0, color="gray", linestyle="--", linewidth=1.5, label="Base Performance"
        )  # Baseline for performance

        plt.legend(title="Version")
        plt.tight_layout()

        # Save the file
        filename = f"performance_{gpu_name}_logdelta_{log_delta}.png"
        plt.savefig(filename)
        print(f"Graph saved to {filename}")
        plt.close()  # Close the current figure to prepare for the next plot


# --- 3. Main Execution Logic ---
if __name__ == "__main__":
    if not os.path.isdir(BUILD_DIR):
        print(f"Fatal Error: Build directory '{BUILD_DIR}' not found.")
        exit(1)

    all_data_records = []

    for version_name, target_path in lib_versions.items():
        print("-" * 60)
        print(f"Setting library version to: {version_name}")
        try:
            symlink_force(target_path, SYMLINK_NAME)
        except Exception:
            exit(1)

        for workload_name, executable in workloads.items():
            print(f"  - Running workload: {workload_name}...")
            times = run_and_parse_workload(executable)

            if times and len(times) == len(LOG_DELTAS):
                for i, time_val in enumerate(times):
                    all_data_records.append(
                        {
                            "Workload": workload_name,
                            "Version": version_name,
                            "log_delta": LOG_DELTAS[i],
                            "Time (us)": time_val,
                        }
                    )
            else:
                count = len(times) if times else 0
                print(
                    f"  -> Warning: Expected {len(LOG_DELTAS)} time values but found {count} for {workload_name}. Skipping."
                )

    df = pd.DataFrame(all_data_records)

    print("\n" + "=" * 60)
    print("Performance Measurement Results (µs) - Pivot Table:")
    if not df.empty:
        pivot_df = df.pivot_table(
            index=["Workload", "log_delta"], columns="Version", values="Time (us)"
        )
        print(pivot_df.to_string())
    else:
        print("No data was collected.")
    print("=" * 60)

    if not df.empty:
        csv_filename = f"performance_results_{NAME}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Detailed results saved to {csv_filename}")

        normalize_and_plot(df, NAME)
    else:
        print("Skipping plotting due to lack of data.")
