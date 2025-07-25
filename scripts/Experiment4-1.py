import subprocess
import pandas as pd
import logging
from pathlib import Path

# --- Configuration ---
# Configure logging to display timestamps, log level, and messages.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define workloads and their corresponding executable names.
WORKLOADS = {
    "ResNet20": ["./resnet_acc", "--num_test_images", "10000"],
}

# Values to be used as column names in the results table.
# This order should match the output order from the executables.
LOG_DELTAS = [30, 35, 40, 48]

# Define key paths as constants using pathlib for robustness.
BASE_DIR = Path("/cheddar")
BUILD_DIR = BASE_DIR / "unittest" / "build"
LIB_DIR = BASE_DIR / "lib"
SYMLINK_TARGET = LIB_DIR / "parfuse" / "libcheddar_parfuse.so"
SYMLINK_PATH = LIB_DIR / "libcheddar.so"
OUTPUT_CSV_FILE = "acc_table.csv"


def update_symlink(target_path: Path, link_path: Path) -> None:
    """
    Safely removes an existing symbolic link and creates a new one.
    """
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        logging.info(f"Creating new symlink: {link_path} -> {target_path}")
        link_path.symlink_to(target_path)
    except OSError as e:
        logging.error(f"Failed to update symlink: {e}")
        raise


def run_workload(name: str, command_parts: list[str]) -> str | None:
    executable = command_parts[0]
    args = command_parts[1:]

    command = [BUILD_DIR / executable] + args

    logging.info(f"Running {name} workload with command: {' '.join(map(str, command))}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=BUILD_DIR)
        if result.returncode != 0:
            logging.warning(
                f"Workload '{name}' exited with a non-zero status ({result.returncode}). Parsing output anyway."
            )
            logging.warning(f"STDERR for '{name}':\n{result.stderr}")
        return result.stdout
    except FileNotFoundError:
        logging.error(f"Executable not found: {command[0]}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while running {name}: {e}")
        return None


def parse_output(output: str) -> list[float]:
    """
    Parses 'Accuracy' or 'precision' values from the workload's output.
    """
    values = []
    if not output:
        return values
    for line in output.splitlines():
        if "Accuracy" in line or "precision" in line:
            try:
                value = float(line.split()[-1])
                values.append(value)
            except (ValueError, IndexError):
                logging.warning(f"Could not parse a valid float from line: '{line}'")
    return values


def print_summary_table(df: pd.DataFrame) -> None:
    """
    Formats and prints the data in a style similar to the provided image.
    """
    if df.empty:
        logging.warning("Cannot print summary table because the DataFrame is empty.")
        return

    # --- Data Transformation for Pivoting ---
    # 1. Melt DataFrame from wide to long format
    long_df = pd.melt(
        df, id_vars=["Workload"], var_name="log_delta_str", value_name="Value"
    )

    # 2. Clean up and convert log_delta column to numeric
    long_df["log_delta"] = long_df["log_delta_str"].str.replace("logΔ=", "").astype(int)

    # 3. Define metrics and clean up workload names for the final table header
    metric_map = {"ResNet20": ("ResNet", "Acc. (%)")}
    long_df[["Workload_Header", "Metric_Header"]] = (
        long_df["Workload"].map(metric_map.get).apply(pd.Series)
    )

    accuracy_workloads = ["ResNet20"]
    long_df.loc[long_df["Workload"].isin(accuracy_workloads), "Value"] *= 100

    # --- Pivoting and Formatting ---
    # 4. Pivot the table to the desired shape with a multi-level header
    try:
        summary_df = long_df.pivot_table(
            index="log_delta",
            columns=["Workload_Header", "Metric_Header"],
            values="Value",
        )
    except Exception as e:
        logging.error(f"Failed to pivot data for summary table. Error: {e}")
        return

    # 5. Sort by delta value in descending order, similar to the image
    summary_df = summary_df.sort_index(ascending=False)

    # 6. Reorder columns to match the image's layout (ResNet)
    summary_df = summary_df[["ResNet"]]

    # 7. Format the index labels to match the image
    delta_format_map = {
        48: "Δ = 2^48 (DR)",
        40: "Δ = 2^40 (RR)",
        35: "Δ = 2^35 (RR)",
        30: "Δ = 2^30 (SR)",
    }
    summary_df.index = summary_df.index.map(delta_format_map)
    summary_df.index.name = None  # Remove index name for cleaner look

    # --- Final Print ---
    print("\n" + "=" * 80)
    print("Functionality Summary")
    print("=" * 80)
    # Use to_string() for a well-formatted console output
    print(summary_df.to_string(float_format="%.2f"))
    print("=" * 80 + "\n")


def main() -> None:
    """
    The main execution function for the script.
    """
    # 1. Set the symbolic link to the optimized library.
    try:
        update_symlink(SYMLINK_TARGET, SYMLINK_PATH)
    except Exception:
        logging.critical("Could not set up the required library symlink. Exiting.")
        return

    # 2. Run all workloads and collect their results.
    results_data = {}
    # Use the order from WORKLOADS dict for consistent processing
    for name, executable in WORKLOADS.items():
        output = run_workload(name, executable)
        if output is not None:
            parsed_values = parse_output(output)
            if parsed_values:
                # Ensure the number of values matches expectations
                if len(parsed_values) == len(LOG_DELTAS):
                    results_data[name] = parsed_values
                else:
                    logging.warning(
                        f"Expected {len(LOG_DELTAS)} values for '{name}', but found {len(parsed_values)}. Skipping."
                    )
            else:
                logging.warning(
                    f"No accuracy/precision values found in the output for workload '{name}'."
                )

    if not results_data:
        logging.error("No data was collected from any workloads. Exiting.")
        return

    # 3. Convert the results into a standard Pandas DataFrame.
    try:
        df = pd.DataFrame.from_dict(
            results_data, orient="index", columns=[f"logΔ={d}" for d in LOG_DELTAS]
        )
        df.insert(0, "Workload", df.index)
        df.reset_index(drop=True, inplace=True)

        # 4. Save the standard results to a CSV file first.
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        logging.info(f"Standard results successfully saved to {OUTPUT_CSV_FILE}")

        # 5. Print the specially formatted summary table to the console.
        print_summary_table(df)

    except Exception as e:
        logging.error(f"Failed to create DataFrame or save file. Error: {e}")
        logging.error(f"Collected data that caused the error: {results_data}")


if __name__ == "__main__":
    main()
