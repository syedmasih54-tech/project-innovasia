"""
StreamShield - Step 10: End-to-End Orchestrator
Executes the full StreamShield pipeline in strict sequence:
1. generator.py -> Makes fake UPI payment data (Round 1 & Round 2)
2. features.py  -> Stateful account memory & feature extraction
3. train.py     -> Model training (LightGBM + Calibration + Isolation Forest)
4. replay.py    -> 2,000 TPS real-time streaming replay benchmark
5. rings.py     -> Graph-based mule ring detection
6. round2.py    -> Concept drift resilience & online recalibration

At the end, prints instructions to launch the live Streamlit dashboard.
"""

import os
import sys
import time
import subprocess

PIPELINE_STEPS = [
    ("Step 1: Generate Synthetic UPI Data", ["python", "src/generator.py"]),
    ("Step 3: Extract Stateful Features",   ["python", "src/features.py"]),
    ("Step 4: Train Hybrid ML Model",       ["python", "src/train.py"]),
    ("Step 6: Real-Time Streaming Replay",  ["python", "src/replay.py"]),
    ("Step 7: Detect Mule Rings (Graph)",   ["python", "src/rings.py"]),
    ("Step 9: Round 2 Drift Adaptation",    ["python", "src/round2.py"])
]


def run_pipeline():
    print("=" * 72)
    print(" 🛡️  STREAMSHIELD END-TO-END PIPELINE ORCHESTRATOR")
    print("=" * 72)
    print("Executing all modules in chronological order...\n")
    
    total_start = time.time()
    python_exec = sys.executable
    
    for step_num, (step_name, cmd) in enumerate(PIPELINE_STEPS, 1):
        print(f"[{step_num}/{len(PIPELINE_STEPS)}] Running {step_name}...")
        step_start = time.time()
        
        # Replace 'python' with current virtual environment interpreter
        exec_cmd = [python_exec] + cmd[1:]
        
        result = subprocess.run(exec_cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n❌ Error in {step_name}! Pipeline stopped.")
            sys.exit(result.returncode)
            
        step_elapsed = time.time() - step_start
        print(f"✓ {step_name} completed in {step_elapsed:.1f}s\n")
        
    total_elapsed = time.time() - total_start
    print("=" * 72)
    print(f"🎉 PIPELINE EXECUTION FINISHED SUCCESSFULLY IN {total_elapsed:.1f}s!")
    print("=" * 72)
    print("\nALL DONE, now run:\n")
    print("    streamlit run app/dashboard.py\n")
    print("=" * 72)


if __name__ == "__main__":
    run_pipeline()

