import subprocess
import sys

def main():
    """
    Runs the ML pipeline scripts in sequence using subprocess.run.
    """
    print("========= Starting ML Pipeline =========")

    # Sequence of scripts to be executed
    scripts = [
        "01_data_ingestion_and_preprocessing.py",
        "02_model_training.py",
        "03_model_prediction.py"
    ]

    python_executable = sys.executable

    for script in scripts:
        print(f"\n>>> Running: {script}")
        
        try:
            subprocess.run([python_executable, script], check=True)
        
        except FileNotFoundError:
            print(f"\n---! Error: Script '{script}' not found. Pipeline halted. !---")
            break
        except subprocess.CalledProcessError:
            print(f"\n---! An error occurred in '{script}'. Pipeline halted. !---")
            break
            
    else:
        print("\n========= ML Pipeline Completed Successfully =========")

if __name__ == "__main__":
    main()