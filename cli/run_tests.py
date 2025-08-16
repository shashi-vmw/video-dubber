#!/usr/bin/env python3
"""
Test runner script for the CLI video dubber.
This script provides convenient commands for running different test suites.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and display the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
    else:
        print(f"❌ {description} failed!")
        
    return result.returncode


def main():
    """Main test runner function."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [command]")
        print("\nAvailable commands:")
        print("  all          - Run all tests")
        print("  unit         - Run only unit tests")
        print("  integration  - Run only integration tests")
        print("  validation   - Run only CLI validation tests")
        print("  config       - Run only configuration tests")
        print("  main         - Run only main CLI function tests")
        print("  coverage     - Run tests with coverage report")
        print("  install      - Install test dependencies")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Change to the CLI directory
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cli_dir)
    
    if command == "install":
        return run_command(
            ["pip", "install", "-r", "requirements-test.txt"],
            "Installing test dependencies"
        )
    
    elif command == "all":
        return run_command(
            ["python", "-m", "pytest", "-v"],
            "Running all tests"
        )
    
    elif command == "unit":
        return run_command(
            ["python", "-m", "pytest", "-v", "-m", "unit"],
            "Running unit tests"
        )
    
    elif command == "integration":
        return run_command(
            ["python", "-m", "pytest", "-v", "-m", "integration"],
            "Running integration tests"
        )
    
    elif command == "validation":
        return run_command(
            ["python", "-m", "pytest", "-v", "tests/test_cli_validation.py"],
            "Running CLI validation tests"
        )
    
    elif command == "config":
        return run_command(
            ["python", "-m", "pytest", "-v", "tests/test_config_creation.py"],
            "Running configuration tests"
        )
    
    elif command == "main":
        return run_command(
            ["python", "-m", "pytest", "-v", "tests/test_main_cli.py"],
            "Running main CLI function tests"
        )
    
    elif command == "coverage":
        return run_command(
            ["python", "-m", "pytest", "--cov=.", "--cov-report=html", "--cov-report=term-missing"],
            "Running tests with coverage report"
        )
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()