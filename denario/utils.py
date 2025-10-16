from .llm import LLM, models
import os, re

def input_check(str_input: str) -> str:
    """Check if the input is a string with the desired content or the path markdown file, in which case reads it to get the content."""

    if str_input.endswith(".md"):
        with open(str_input, 'r') as f:
            content = f.read()
    elif isinstance(str_input, str):
        content = str_input
    else:
        raise ValueError("Input must be a string or a path to a markdown file.")
    return content

def llm_parser(llm: LLM | str) -> LLM:
    """Get the LLM instance from a string."""
    if isinstance(llm, str):
        try:
            llm = models[llm]
        except KeyError:
            raise KeyError(f"LLM '{llm}' not available. Please select from: {list(models.keys())}")
    return llm

def extract_file_paths(markdown_text):
    """
    Extract the bulleted file paths from markdown text 
    and check if they exist and are absolute paths.
    
    Args:
        markdown_text (str): The markdown text containing file paths
    
    Returns:
        tuple: (existing_paths, missing_paths)
    """
    
    # Pattern to match file paths in markdown bullet points
    pattern = r'-\s*([^\n]+\.(?:csv|txt|md|py|json|yaml|yml|xml|html|css|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|php|rb|pl|sh|bat|sql|log))'
    
    # Find all matches
    matches = re.findall(pattern, markdown_text, re.IGNORECASE)
    
    # Clean up paths and check existence
    existing_paths = []
    missing_paths = []
    
    for match in matches:
        path = match.strip()
        if os.path.exists(path) and os.path.isabs(path):
            existing_paths.append(path)
        else:
            missing_paths.append(path)
    
    return existing_paths, missing_paths
