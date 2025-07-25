import subprocess
import pandas as pd
import logging
from pathlib import Path

# --- Configuration ---
# Configure logging to display timestamps, log level, and messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define workloads and their corresponding executable names.
WORKLOADS = {
    "ResNet20": "./resnet",
    "Sorting": "./sorting",
    "Bts": "./boot_test",
    "HELR": "./helr",
}

# Values to be used as column names in the results table.
LOG_DELTAS = [30, 35, 40, 48]

# Define key paths as constants using pathlib for robustness.
BASE_DIR = Path("/cheddar")
BUILD_DIR = BASE_DIR / "unittest" / "build"
LIB_DIR = BASE_DIR / "lib"
SYMLINK_TARGET = LIB_DIR / "parfuse" / "libcheddar_parfuse.so"
SYMLINK_PATH = LIB_DIR / "libcheddar.so"
OUTPUT_CSV_FILE = "execution_times_table.csv"


def update_symlink(target_path: Path, link_path: Path) -> None:
    """
    Safely removes an existing symbolic link and creates a new one.

    Args:
        target_path (Path): The path to the actual library file the link should point to.
        link_path (Path): The path of the symbolic link to create.
    """
    try:
        # Remove the link if it exists (as a link or a file).
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        
        logging.info(f"Creating new symlink: {link_path} -> {target_path}")
        link_path.symlink_to(target_path)
    except OSError as e:
        logging.error(f"Failed to update symlink: {e}")
        raise


def run_workload(name: str, executable: str) -> str | None:
    """
    Runs a specified workload and returns its standard output.
    This function will return the output even if the process has a non-zero exit code.

    Args:
        name (str): The name of the workload (for logging).
        executable (str): The name of the executable file within the build directory.

    Returns:
        str | None: The standard output string, or None if the command fails to run.
    """
    command = [BUILD_DIR / executable]
    logging.info(f"Running {name} workload with command: {' '.join(map(str, command))}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=BUILD_DIR
        )
        # Log a warning for non-zero exit codes but still proceed with the output.
        if result.returncode != 0:
            logging.warning(f"Workload '{name}' exited with a non-zero status ({result.returncode}).")
            if name == "Sorting":
                logging.warning(f"Small delta (Δ = 2^30) failure in Sorting workload is an expected result.")
                logging.warning(f"You can ignore this warning.") 
            else:
                logging.warning(f"STDERR for '{name}':\n{result.stderr}")

            
        return result.stdout
    except FileNotFoundError:
        logging.error(f"Executable not found: {command[0]}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while running {name}: {e}")
        return None


def parse_execution_times(output: str) -> list[float]:
    """
    Parses 'Wall clock time' values from the workload's output.

    Args:
        output (str): The standard output from the workload.

    Returns:
        list[float]: A list of the extracted float values in microseconds.
    """
    times = []
    if not output:
        return times
        
    for line in output.splitlines():
        if "Wall clock time" in line:
            parts = line.split()
            time_part = parts[-1]
            try:
                # Remove 'us' suffix and convert to float
                time_value = float(time_part[:-2])
                times.append(time_value)
            except (ValueError, IndexError):
                logging.warning(f"Could not parse time value from line: '{line}'")
    return times


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

    # 2. Run all workloads and collect their execution times.
    execution_times = {}
    for name, executable in WORKLOADS.items():
        output = run_workload(name, executable)
        if output is None:
            continue
        
        times = parse_execution_times(output)
        if times:
            execution_times[name] = times
        else:
            logging.warning(f"No execution times found for workload '{name}'.")

    if not execution_times:
        logging.error("No execution times were collected from any workloads. Exiting.")
        return

    # 3. Convert the results into a Pandas DataFrame.
    try:
        # Dynamically create columns to avoid errors if output format changes.
        df = pd.DataFrame.from_dict(
            execution_times,
            orient='index',
            columns=[f"logΔ={d}" for d in LOG_DELTAS]
        )
        df.insert(0, "Workload", df.index)
        df.reset_index(drop=True, inplace=True)

        # 4. Print the results and save them to a CSV file.
        print("\n" + "="*80)
        print("Execution times (in us) for each workload and configuration:")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

        df.to_csv(OUTPUT_CSV_FILE, index=False)
        logging.info(f"Execution times successfully saved to {OUTPUT_CSV_FILE}")

    except Exception as e:
        logging.error(f"Failed to create or save the DataFrame. Error: {e}")
        logging.error(f"Collected data that caused the error: {execution_times}")


if __name__ == "__main__":
    main()
