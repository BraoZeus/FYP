"""
SAFELABS Interactive Visual Test Suite Launcher
Provides a unified launcher to run any non-functional or unit test individually,
or execute the full visual test suite sequentially with interactive GUI windows.

Run via:
    python tests/run_all_visual_tests.py
"""

import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT_DIR / "tests"

TEST_SCRIPTS = [
    ("Pipeline Runtime & Latency Benchmark", "test_pipeline_runtime.py", "Non-Functional (Performance)"),
    ("Stream Ingestion & Watchdog Recovery", "test_stream_ingestion.py", "Non-Functional (Reliability)"),
    ("Memory Footprint & Leak Prevention", "test_memory_footprint.py", "Non-Functional (Scalability)"),
    ("Cryptographic Security & RBAC Integrity", "test_security_cryptography.py", "Non-Functional (Security)"),
    ("Algorithmic & Computer Vision Unit Tests", "test_algorithmic_units.py", "Unit Tests (Algorithms)"),
]

def run_single_test(script_name: str, desc: str):
    script_path = TESTS_DIR / script_name
    print(f"\n================================================================================")
    print(f" LAUNCHING: {desc}")
    print(f" Script: {script_path}")
    print(f" Note: An interactive GUI window will open. Press any key in the window when done.")
    print(f"================================================================================\n")
    res = subprocess.run([sys.executable, str(script_path)])
    return res.returncode

def main():
    while True:
        print("\n" + "=" * 80)
        print(" SAFELABS INTERACTIVE VISUAL TEST SUITE RUNNER")
        print(" Each selection opens an interactive window displaying live test execution.")
        print("=" * 80)
        for idx, (title, script, category) in enumerate(TEST_SCRIPTS, 1):
            print(f" [{idx}] {title:<42} [{category}]")
        print(f" [6] Run All 5 Tests Sequentially")
        print(f" [0] Exit")
        print("=" * 80)

        choice = input("Enter choice [0-6]: ").strip()
        if choice == "0":
            print("Exiting test runner.")
            break
        elif choice in [str(i) for i in range(1, 6)]:
            idx = int(choice) - 1
            title, script, _ = TEST_SCRIPTS[idx]
            run_single_test(script, title)
        elif choice == "6":
            print("\nStarting full sequential visual test suite...")
            for title, script, _ in TEST_SCRIPTS:
                rc = run_single_test(script, title)
                if rc != 0:
                    print(f"[WARN] Test {script} exited with code {rc}")
            print("\n================================================================================")
            print(" All tests completed! All visual screenshots saved to tests/visual_evidence/")
            print("================================================================================\n")
        else:
            print("Invalid selection. Please enter 0-6.")

if __name__ == "__main__":
    main()

