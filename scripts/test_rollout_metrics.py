
import sys
import json
import subprocess
import shutil
from pathlib import Path

def test_rollout_metrics():
    """
    Regression test for Rollout-Light metrics.
    Runs run_edgepatch.py on a single example and verifies nonzero scores.
    """
    repo_root = Path(__file__).parent.parent
    scripts_dir = repo_root / "scripts"
    output_dir = repo_root / "debug_rollout_metric"
    
    # Clean previous output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create regression config to avoid CLI boolean parsing issues and force CPU/GPT2
    config_path = output_dir / "regression_config.yaml"
    with open(config_path, "w") as f:
        f.write("""
model_name: "gpt2"
load_in_4bit: false
dataset_name: "HuggingFaceH4/MATH-500"
dataset_streaming: true
max_examples: 1
verbose: true
rollout_k: 2
rollout_h: 32
max_decision_points: 2
""")

    cmd = [
        sys.executable,
        str(scripts_dir / "run_edgepatch.py"),
        "main",  # Mode
        "--config", str(config_path),
        "--method", "rollout_light",
        "--example-ids", "problem_1591",
        "--output-dir", str(output_dir),
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd, cwd=repo_root)
    except subprocess.CalledProcessError as e:
        print(f"Run failed with code {e.returncode}")
        # Even if it failed (e.g. example not found), we check why
        # If it failed due to fail-fast, that's interesting too
        # But we expect success.
        sys.exit(1)
        
    # Check results
    # run_dir is likely inside output_dir with timestamp
    # Find the latest run dir
    run_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    if not run_dirs:
        print("No run directory created!")
        sys.exit(1)
        
    run_dir = run_dirs[-1]
    results_file = run_dir / "all_results.json"
    
    if not results_file.exists():
        print(f"all_results.json not found in {run_dir}")
        sys.exit(1)
        
    print(f"Loading results from {results_file}...")
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("all_results.json is empty!")
        sys.exit(1)
        
    example_res = data[0]
    scores = example_res.get("scores", [])
    
    if not scores:
        print("No scores found for example!")
        sys.exit(1)
        
    # Validation Rules
    # 1. Check nonzero scores
    nonzero_delta = sum(1 for s in scores if abs(s["delta_logp"]) > 1e-6)
    nonzero_mask = sum(1 for s in scores if s["masked_entries_count"] > 0)
    
    print(f"Total chunks: {len(scores)}")
    print(f"Chunks with nonzero delta_logp: {nonzero_delta}")
    print(f"Chunks with masked_entries_count > 0: {nonzero_mask}")
    
    details = example_res.get("method_details", {})
    rollouts = details.get("rollout_results", [])
    print(f"Rollout sets: {len(rollouts)}")
    
    if len(rollouts) == 0:
        print("FAIL: No rollouts performed!")
        sys.exit(1)
        
    for i, r in enumerate(rollouts):
        print(f"Rollout {i}: delta_logp={r['delta_logp']}, masks={r['masked_entries_count']}")
        if r['masked_entries_count'] == 0:
            print(f"FAIL: Rollout {i} has 0 masked entries!")
            sys.exit(1)
            
    if nonzero_delta == 0:
        print("FAIL: All scores have zero delta_logp!")
        sys.exit(1)
        
    if nonzero_mask == 0:
        print("FAIL: All scores have zero masked_entries_count!")
        sys.exit(1)
        
    print("SUCCESS: Rollout metrics verified non-zero.")

if __name__ == "__main__":
    test_rollout_metrics()
