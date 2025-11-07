import sys
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(prog="denario")
    subparsers = parser.add_subparsers(dest="command")

    # `denario run`
    subparsers.add_parser("run", help="Run the Denario Streamlit app")

    args = parser.parse_args()

    if args.command == "run":
        try:
            import streamlit
            # Construct the full path to the app.py file
            app_path = os.path.join(os.path.dirname(__file__), 'app.py')
            subprocess.Popen(["streamlit", "run", app_path])
        except ImportError:
            print("❌ Streamlit not installed. Install with: pip install streamlit")
            sys.exit(1)
    else:
        parser.print_help()
