import subprocess
import sys

def run_diagnostics():
    print("Launching trading bot in monitored diagnostic mode...")
    
    # Run python run_trader.py as a subprocess
    process = subprocess.Popen(
        [sys.executable, "run_trader.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Read output line by line as it prints
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"[BOT-STDOUT] {output.strip()}")
            
    # Capture any raw stderr
    stdout, stderr = process.communicate()
    
    print("\n--- DIAGNOSTIC RESULT ---")
    print(f"Subprocess Exit Code: {process.returncode}")
    if stderr:
        print(f"System Error Output (stderr):\n{stderr}")
    else:
        print("No stderr output captured.")
        
if __name__ == "__main__":
    run_diagnostics()
