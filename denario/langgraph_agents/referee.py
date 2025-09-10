from langchain_core.runnables import RunnableConfig

from ..paper_agents.tools import extract_latex_block, LLM_call_stream, clean_section
from .prompts import reviewer_fast_prompt
from .parameters import GraphState


def referee_fast(state: GraphState, config: RunnableConfig):

    print('Reviewing the paper...', end="", flush=True)

    PROMPT = reviewer_fast_prompt(state)
    state, result = LLM_call_stream(PROMPT, state)
    text = extract_latex_block(state, result, "REVIEW")

    # remove LLM added lines
    text = clean_section(text, "REVIEW")

    with open(state['files']['referee_report'], 'w') as f:
        f.write(text)

    print(f"done {state['tokens']['ti']} {state['tokens']['to']}")
