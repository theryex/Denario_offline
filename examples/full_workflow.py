# This script runs the full workflow of Denario.
# and allows specifying the LLM models to use.
# The 'mini' models are much cheaper than the 'pro' models.
# but their output quality can be slightly lower.
# see e.g., https://platform.openai.com/docs/pricing

from denario import Denario, Journal

# This is one of the example projects in the examples folder.
# It investigates the properties of a recent gravitational wave detection
# called GW231123 by the LIGO and Virgo detectors (https://arxiv.org/pdf/2507.08219)
# The data description is in the input.md file.
# All the outputs are saved in the same folder.
folder = "GW231123"
astro_pilot = Denario(project_dir=folder)

# Set the input prompt containing the data description
# WARNING: PLEASE PROVIDE ABSOLUTE PATHS to all the data files listed in the .md file
# (otherwise this may cause hallucinations in the LLMs)
astro_pilot.set_data_description(f"{folder}/input.md")

# This module generates the idea to be investigated.
# get_idea() allows to employ two backends: a planning and control workflow from cmbagent or a faster method based on Langgraph
# get_idea(mode="fast") is a faster version than get_idea(mode="cmbagent") 
# but can produce results with slightly lower quality.
# same logic below for get_method()
astro_pilot.get_idea(mode="fast",llm='gpt-4.1-mini') 

# This module checks if the idea is novel or not against previous literature
astro_pilot.check_idea(llm='gpt-4.1-mini', max_iterations=7) 

# This module generates the methodology to be employed.
astro_pilot.get_method(mode="fast",llm='gpt-4.1-mini') 

# This module writes codes, executes the codes, makes plots,
#  and summarizes the results.
astro_pilot.get_results(engineer_model='gpt-4.1-mini',
                        researcher_model='gpt-4.1-mini',
                        planner_model='gpt-4.1-mini',
                        plan_reviewer_model='gpt-4.1-mini',
                        orchestration_model='gpt-4.1-mini',
                        formatter_model='gpt-5-mini',
                        )

# Get the paper
astro_pilot.get_paper(journal=Journal.AAS, llm='gpt-4.1-mini', add_citations=False) 

# Referee the paper
astro_pilot.referee(llm='gpt-4.1-mini')
