import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from denario.denario import Denario

if __name__ == "__main__":
    denario = Denario()
    denario.render_ui()
