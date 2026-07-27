import subprocess
import sys

def run_script(script_name):
    print("=" * 60)
    print(f"🚀 Running {script_name}...")
    print("=" * 60)
    # Run the script with the current Python interpreter
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"{script_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"{script_name} completed successfully.\n")

if __name__ == "__main__":
    try:
        run_script("fetch_dynamic_instructions.py")
        run_script("agent_evaluation_script.py")
        print("All scripts ran successfully!")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
