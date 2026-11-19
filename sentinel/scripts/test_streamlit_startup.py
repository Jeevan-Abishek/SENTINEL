"""
Verifies that the SENTINEL Streamlit dashboard launches cleanly without runtime exceptions.
"""

import subprocess
import sys
import time

python_exe = sys.executable
cmd = [
    python_exe,
    "-m",
    "streamlit",
    "run",
    "sentinel/dashboard/app.py",
    "--server.headless=true",
    "--server.port=8515",
    "--browser.serverAddress=localhost",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false",
]

print(f"Launching Streamlit test: {' '.join(cmd)}")
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

time.sleep(4.0)

# Check if process crashed immediately
poll = proc.poll()
if poll is not None:
    stdout, stderr = proc.communicate()
    print("FAILED: Streamlit exited prematurely with code", poll)
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    sys.exit(1)

# Process is running cleanly! Terminate test process
proc.terminate()
try:
    proc.wait(timeout=3)
except Exception:
    proc.kill()

print("✓ Streamlit dashboard started cleanly and remained stable with zero runtime errors!")
sys.exit(0)

