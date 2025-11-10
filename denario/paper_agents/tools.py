import re
import sys
import json
import json5
from pathlib import Path

from .prompts import fixer_prompt, LaTeX_prompt
from .parameters import GraphState
from .journal import LatexPresets
from .latex_presets import journal_dict


def LLM_call(prompt, state):
    """
    This function calls the LLM and update tokens
    """

    # Resolve the runnable LLM client from the state. The state['llm'] structure
    # may be either:
    # - a dict that contains a runtime client under key 'llm' (expected), or
    # - already the runtime client object (rare), or
    # - a higher-level LLM config object (missing runtime client).
    llm_entry = state.get('llm')
    if llm_entry is None:
        raise KeyError("LLM entry missing from state. Ensure the graph preprocess node ran and attached an LLM client under state['llm']['llm'].")

    # prefer the runtime client if available
    runner = None
    if isinstance(llm_entry, dict):
        runner = llm_entry.get('llm') or llm_entry
    else:
        runner = llm_entry

    # Runner must expose .invoke
    if not hasattr(runner, 'invoke'):
        raise KeyError("No runnable LLM client found at state['llm']['llm'] (object has no .invoke).")

    message = runner.invoke(prompt)

    # Safely read usage metadata
    usage = getattr(message, 'usage_metadata', {}) or {}
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    max_output = None
    if isinstance(llm_entry, dict):
        max_output = llm_entry.get('max_output_tokens')

    if max_output is not None and output_tokens > max_output:
        print('WARNING!! Max output tokens reach!')

    state['tokens']['ti'] += input_tokens
    state['tokens']['to'] += output_tokens
    state['tokens']['i'] = input_tokens
    state['tokens']['o'] = output_tokens
    with open(state['files']['LLM_calls'], 'a') as f:
        f.write(f"{state['tokens']['i']} {state['tokens']['o']} {state['tokens']['ti']} {state['tokens']['to']}\n")

    return state, getattr(message, 'content', message)


def LLM_call_stream(prompt, state):
    """
    Handles both streaming and non-streaming LLM calls, writing output to file.
    For non-streaming clients (like vLLM/Ollama), fetches complete response and writes it at once.
    For streaming clients, writes chunks as they arrive.
    """
    output_file_path = state['files']['f_stream']

    # Resolve the runnable LLM client from the state. The state['llm'] structure
    # may be either:
    # - a dict that contains a runtime client under key 'llm' (expected), or
    # - already the runtime client object (rare), or
    # - a higher-level LLM config object (missing runtime client).
    llm_entry = state.get('llm')
    if llm_entry is None:
        raise KeyError("LLM entry missing from state. Ensure the graph preprocess node ran and attached an LLM client under state['llm']['llm'].")

    # prefer the runtime client if available
    runner = None
    if isinstance(llm_entry, dict):
        runner = llm_entry.get('llm') or llm_entry
    else:
        runner = llm_entry

    # Extract additional config
    if isinstance(llm_entry, dict):
        stream_verbose = llm_entry.get('stream_verbose', False)
        max_output = llm_entry.get('max_output_tokens')
    else:
        stream_verbose = getattr(llm_entry, 'stream_verbose', False)
        max_output = getattr(llm_entry, 'max_output_tokens', None)


    # If the runner doesn't support streaming, attempt fallback methods
    if not hasattr(runner, 'stream'):
        # First try using synchronous LLM_call if .invoke is available
        if hasattr(runner, 'invoke'):
            try:
                return LLM_call(prompt, state)
            except Exception as e:
                # Log error but continue to other fallbacks
                print(f"Warning: LLM_call failed ({str(e)}), trying alternative methods")

        # Try .generate if available (common for vLLM/Ollama)
        if hasattr(runner, 'generate'):
            try:
                msg = runner.generate(prompt)
                # Extract response text from various possible locations
                text = getattr(msg, 'content', None) or getattr(msg, 'text', None) or str(msg)
                
                # Get usage stats if available
                usage = getattr(msg, 'usage_metadata', {}) or {}
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)

                # Check output length
                if max_output is not None and output_tokens > max_output:
                    print('WARNING!! Max output tokens reached!')

                # Update token counters
                state['tokens']['ti'] += input_tokens
                state['tokens']['to'] += output_tokens
                state['tokens']['i'] = input_tokens
                state['tokens']['o'] = output_tokens

                # Log token usage
                with open(state['files']['LLM_calls'], 'a', encoding='utf-8') as f:
                    f.write(f"{state['tokens']['i']} {state['tokens']['o']} {state['tokens']['ti']} {state['tokens']['to']}\n")

                return state, text
            except Exception as e:
                print(f"Warning: generate() fallback failed ({str(e)})")

        # No working fallback found
        raise KeyError("No streaming LLM client found and all fallback methods failed. The client must expose either .stream, .invoke, or .generate methods.")

    # Initialize streaming state
    full_content = ''
    total_input_tokens = 0
    total_output_tokens = 0

    # Start streaming and writing output
    with open(output_file_path, 'a', encoding='utf-8') as f:
        try:
            for chunk in runner.stream(prompt):
                # Extract text content (may be string or object)
                text = getattr(chunk, 'content', chunk)
                if not isinstance(text, str):
                    text = str(text)

                # Write to file and optionally console
                f.write(text)
                f.flush()  # Force immediate write
                if stream_verbose:
                    print(text, end='', flush=True)
                full_content += text

                # Track token usage from chunk metadata if available
                usage = getattr(chunk, 'usage_metadata', {}) or {}
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                
                # Accumulate tokens
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                # Check max tokens if specified
                if max_output is not None and total_output_tokens > max_output:
                    print('WARNING!! Max output tokens reached!')

            # Add final newlines
            f.write('\n\n')

        except Exception as e:
            print(f"Warning: Streaming error occurred ({str(e)})")
            raise

    # Update final token counts in state
    state['tokens']['i'] = total_input_tokens
    state['tokens']['o'] = total_output_tokens
    state['tokens']['ti'] += total_input_tokens
    state['tokens']['to'] += total_output_tokens

    # Log token usage
    with open(state['files']['LLM_calls'], 'a', encoding='utf-8') as f:
        f.write(f"{state['tokens']['i']} {state['tokens']['o']} {state['tokens']['ti']} {state['tokens']['to']}\n")

    return state, full_content



def temp_file(state, fin, action, text=None, json_file=False):
    """
    This function reads or writes the content of a temporary file
    fin:  the name of the file
    action: either 'read' of 'write'
    text: when action is 'write', the text to write
    json: whether the file is json or not
    """
    
    journaldict: LatexPresets = journal_dict[state['paper']['journal']]

    if action=='read':
        with open(fin, 'r', encoding='utf-8') as f:
            if json_file:
                return json.load(f)
            else:
                latex_text = f.read()
                
                # Extract content between \begin{document} and \end{document}
                match = re.search(r'\\begin{document}(.*?)\\end{document}',
                                  latex_text, re.DOTALL)

                if match:
                    extracted_text = match.group(1).strip()
                    return extracted_text
                else:
                    raise Exception("Text not found on file!")

    elif action=='write':
        with open(fin, 'w', encoding='utf-8') as f:
            if json_file:
                json.dump(text, f, indent=2)
            else:
                latex_text = rf"""\documentclass[{journaldict.layout}]{{{journaldict.article}}}

\usepackage{{amsmath}}
\usepackage{{multirow}}
\usepackage{{natbib}}
\usepackage{{graphicx}} 
{journaldict.usepackage}

\begin{{document}}

{text}

\end{{document}}
                """
                f.write(latex_text)
    else:
        raise Exception("wrong action chosen!")


def json_parser(text):
    """
    This function extracts the text between ```json ```
    """
    
    json_pattern = r"```json(.*)```"
    match = re.findall(json_pattern, text, re.DOTALL)
    json_string = match[0].strip()
    json_string = json_string.replace("\\", "\\\\") #deal with unescaped backslashes
    try:
        parsed_json = json.loads(json_string)
    except json.decoder.JSONDecodeError:
        try:
            json_string = json_string.replace("'", "\"")
            parsed_json = json.loads(json_string)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    return parsed_json



def json_parser2(text: str):
    """
    Extract the first ```json … ``` fenced block and parse it.
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        # fallback: any fenced block
        m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON fenced block found.")

    json_string = m.group(1)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Helpful error to see exactly where it failed
        snippet = json_string[max(0, e.pos-40):e.pos+40]
        raise ValueError(f"JSON parse error at pos {e.pos}: {e.msg}\n…{snippet}…")

    
def json_parser3(text: str):
    """
    This function extracts a json data from a text
    """
    
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        # fallback: any fenced block
        m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        new_text = f"```json\n{text}\n```"
        m = re.search(r"```json\s*(\{.*?\})\s*```", new_text, re.DOTALL | re.IGNORECASE)
        if not m:
            raise ValueError("No JSON fenced block found.")
    json_string = m.group(1)
    data = json5.loads(json_string)
    return data


def extract_latex_block(state: GraphState, text: str, block: str) -> str:
    r"""
    This function takes some text and extracts the TEXT located between
    \begin{block}
    TEXT
    \end{block}
    """

    # Check if the input 'text' is a list and convert it to a string
    if isinstance(text, list):
        # Join the list items into a single string
        # Use str(item) to ensure all list elements can be joined
        text = "".join([str(item) for item in text])

    pattern = rf"\\begin{{{block}}}(.*?)\\end{{{block}}}"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    
    # in case it fails
    with open(state['files']['Error'], 'w', encoding='utf-8') as f:
        f.write(text)

    # try to fix it using fixed
    try:
        return fixer(state, block)
    except ValueError:
        raise ValueError(f"Failed to extract {block}")

    

def fixer(state: GraphState, section_name):
    """
    This function will try to fix the errors with automatic parsing
    """

    path = Path(state['files']['Error'])
    with path.open("r", encoding="utf-8") as f:
        Text = f.read()
    
    PROMPT = fixer_prompt(Text, section_name)
    state, result = LLM_call(PROMPT, state)
    #result = llm.invoke(PROMPT).content
    
    # Extract caption
    pattern = rf"\\begin{{{section_name}}}(.*?)\\end{{{section_name}}}"
    match = re.search(pattern, result, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        with open(state['files']['Error'], 'w', encoding='utf-8') as f:
            f.write(result)
        print("Fixer failed to extract block")
        sys.exit()



def LaTeX_checker(state, text):

    PROMPT = LaTeX_prompt(text)
    state, result = LLM_call(PROMPT, state)
    #result = llm.invoke(PROMPT).content
    text = extract_latex_block(state, result, "Text")
    return text


def clean_section(text, section):
    """
    This function performs some clean up of unwanted LaTeX wrappers
    """

    text = text.replace(r"\documentclass{article}", "")
    text = text.replace(r"\begin{document}", "")
    text = text.replace(r"\end{document}", "")
    text = text.replace(fr"\section{{{section}}}", "")
    text = text.replace(fr"\section*{{{section}}}", "")
    text = text.replace(fr"\begin{{{section}}}", "")
    text = text.replace(fr"\end{{{section}}}", "")
    text = text.replace(r"\maketitle", "")
    text = text.replace(r"<PARAGRAPH>", "")
    text = text.replace(r"</PARAGRAPH>", "")
    text = text.replace(r"</{section}>", "")
    text = text.replace(r"<{section}>", "")
    text = text.replace(r"```latex", "")
    text = text.replace(r"```", "")
    text = text.replace(r"\usepackage{amsmath}", "")

    return text


def check_images_in_text(state, images):
    """
    This function checks whether the LLM has put the images in the text or not
    """

    # Check that the images are in the text
    for key, value in images.items():
        if value["name"] not in state['paper']['Results']:
            return False
    return True

