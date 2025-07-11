import subprocess
import pandas as pd
import logging
from pathlib import Path

# --- Configuration ---
# Configure logging to display timestamps, log level, and messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define key paths as constants using pathlib for robustness.
BASE_DIR = Path("/cheddar")
BUILD_DIR = BASE_DIR / "unittest" / "build"
LIB_DIR = BASE_DIR / "lib"
SYMLINK_TARGET = LIB_DIR / "parfuse" / "libcheddar_parfuse.so"
SYMLINK_PATH = LIB_DIR / "libcheddar.so"

# Script-specific settings
EXECUTABLE_NAME = "./basic_test"
MECHANISMS = ["HAdd", "HMult", "HRot", "Rescale"]
LIMBS = [24, 48]
OUTPUT_CSV_FILE = "mechanism_time_table.csv"


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


def run_fhe_mechanism_test() -> str | None:
    """
    Runs the FHE mechanism test executable and returns its standard output.

    Returns:
        str | None: The standard output string on success, or None on failure.
    """
    command = [BUILD_DIR / EXECUTABLE_NAME]
    logging.info(f"Running FHE mechanism test with command: {' '.join(map(str, command))}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Raise an exception for non-zero exit codes.
            cwd=BUILD_DIR
        )
        return result.stdout
    except FileNotFoundError:
        logging.error(f"Executable not found: {command[0]}")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running test:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}")
        # Return the output even on error, in case partial data is useful.
        return e.stdout
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None


def parse_mechanism_times(output: str, mechanisms: list[str]) -> dict[str, list[float]]:
    """
    Parses 'Wall clock time' from the output based on the expected order of mechanisms.

    Args:
        output (str): The standard output from the test executable.
        mechanisms (list[str]): The list of mechanism names in the expected order.

    Returns:
        dict[str, list[float]]: A dictionary mapping each mechanism to its execution times.
    """
    execution_times = {mech: [] for mech in mechanisms}
    time_values = []
    
    if not output:
        return {}

    # First, extract all time values from the output.
    for line in output.splitlines():
        if "Wall clock time" in line:
            parts = line.split()
            time_part = parts[-1]
            try:
                # Remove 'us' suffix and convert to float
                value = float(time_part[:-2])
                time_values.append(value)
            except (ValueError, IndexError):
                logging.warning(f"Could not parse time value from line: '{line}'")

    # Then, distribute the collected time values into the mechanism dictionary.
    # This assumes the times appear in pairs for each mechanism, in order.
    time_index = 0
    for mech in mechanisms:
        for _ in range(len(LIMBS)): # Assumes one time value per limb value
            if time_index < len(time_values):
                execution_times[mech].append(time_values[time_index])
                time_index += 1
            else:
                logging.warning(f"Not enough time values in output to fully populate mechanism '{mech}'.")
                break
    
    return execution_times


def main() -> None:
    """
    The main execution function for the script.
    """
    # 1. Set up the required symbolic link.
    try:
        update_symlink(SYMLINK_TARGET, SYMLINK_PATH)
    except Exception:
        logging.critical("Could not set up the required library symlink. Exiting.")
        return

    # 2. Run the test and parse the output.
    output = run_fhe_mechanism_test()
    if output is None:
        logging.error("Failed to get any output from the test executable. Exiting.")
        return
        
    execution_times = parse_mechanism_times(output, MECHANISMS)
    
    # Filter out mechanisms with incomplete data to prevent DataFrame errors.
    valid_data = {
        mech: times for mech, times in execution_times.items() if len(times) == len(LIMBS)
    }

    if not valid_data:
        logging.error("No complete mechanism timing data was collected. Exiting.")
        logging.info(f"Partially collected data: {execution_times}")
        return

    # 3. Convert the results into a Pandas DataFrame.
    try:
        df = pd.DataFrame.from_dict(
            valid_data,
            orient='index',
            columns=[f"limbs={n}" for n in LIMBS]
        )
        df.insert(0, "Mechanism", df.index)
        df.reset_index(drop=True, inplace=True)

        # 4. Print the results and save them to a CSV file.
        print("\n" + "="*80)
        print("FHE Mechanism Execution Times (in us):")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

        df.to_csv(OUTPUT_CSV_FILE, index=False)
        logging.info(f"Mechanism timing data successfully saved to {OUTPUT_CSV_FILE}")

    except Exception as e:
        logging.error(f"Failed to create or save the DataFrame. Error: {e}")
        logging.error(f"Data that caused the error: {valid_data}")


if __name__ == "__main__":
    main()
