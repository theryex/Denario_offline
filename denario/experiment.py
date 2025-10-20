from typing import List
import os
import re
import cmbagent

from .key_manager import KeyManager
from .prompts.experiment import experiment_planner_prompt, experiment_engineer_prompt, experiment_researcher_prompt

class Experiment:
    """
    This class is used to perform the experiment.
    TODO: improve docstring
    """

    def __init__(self,
                 research_idea: str,
                 methodology: str,
                 keys: KeyManager,
                 involved_agents: List[str] = ['engineer', 'researcher'],
                 engineer_model: str = "gpt-4.1",
                 researcher_model: str = "o3-mini-2025-01-31",
                 planner_model: str = "gpt-4o",
                 plan_reviewer_model: str = "o3-mini",
                 work_dir = None,
                 restart_at_step: int = -1,
                 hardware_constraints: str = None,
                 max_n_attempts: int = 10,
                 max_n_steps: int = 6,
                 orchestration_model = "gpt-4.1",
                 formatter_model = "o3-mini",
                ):
        
        self.engineer_model = engineer_model
        self.researcher_model = researcher_model
        self.planner_model = planner_model
        self.plan_reviewer_model = plan_reviewer_model
        self.restart_at_step = restart_at_step
        self.hardware_constraints = hardware_constraints
        self.max_n_attempts = max_n_attempts
        self.max_n_steps = max_n_steps
        self.orchestration_model = orchestration_model
        self.formatter_model = formatter_model

        
        if work_dir is None:
            raise ValueError("workdir must be provided")

        self.api_keys = keys

        self.experiment_dir = os.path.join(work_dir, "experiment_generation_output")
        # Create directory if it doesn't exist
        os.makedirs(self.experiment_dir, exist_ok=True)

        involved_agents_str = ', '.join(involved_agents)

        # Set prompts
        self.planner_append_instructions = experiment_planner_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
            involved_agents_str = involved_agents_str
        )
        self.engineer_append_instructions = experiment_engineer_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
        )
        self.researcher_append_instructions = experiment_researcher_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
        )

    def run_experiment(self, data_description: str, **kwargs):
        """
        Run the experiment.
        TODO: improve docstring
        """

        print(f"Engineer model: {self.engineer_model}")
        print(f"Researcher model: {self.researcher_model}")
        print(f"Planner model: {self.planner_model}")
        print(f"Plan reviewer model: {self.plan_reviewer_model}")
        print(f"Max n attempts: {self.max_n_attempts}")
        print(f"Max n steps: {self.max_n_steps}")
        print(f"Restart at step: {self.restart_at_step}")
        print(f"Hardware constraints: {self.hardware_constraints}")

        results = cmbagent.planning_and_control_context_carryover(data_description,
                            n_plan_reviews = 1,
                            max_n_attempts = self.max_n_attempts,
                            max_plan_steps = self.max_n_steps,
                            max_rounds_control = 500,
                            engineer_model = self.engineer_model,
                            researcher_model = self.researcher_model,
                            planner_model = self.planner_model,
                            plan_reviewer_model = self.plan_reviewer_model,
                            plan_instructions=self.planner_append_instructions,
                            researcher_instructions=self.researcher_append_instructions,
                            engineer_instructions=self.engineer_append_instructions,
                            work_dir = self.experiment_dir,
                            api_keys = self.api_keys,
                            restart_at_step = self.restart_at_step,
                            hardware_constraints = self.hardware_constraints,
                            default_llm_model = self.orchestration_model,
                            default_formatter_model = self.formatter_model
                            )
        chat_history = results['chat_history']
        final_context = results['final_context']
        
        try:
            for obj in chat_history[::-1]:
                if obj['name'] == 'researcher_response_formatter':
                    result = obj['content']
                    break
            task_result = result
        except:
            task_result = None
            
        MD_CODE_BLOCK_PATTERN = r"```[ \t]*(?:markdown)[ \t]*\r?\n(.*)\r?\n[ \t]*```"
        extracted_results = re.findall(MD_CODE_BLOCK_PATTERN, task_result, flags=re.DOTALL)[0]
        clean_results = re.sub(r'^<!--.*?-->\s*\n', '', extracted_results)
        self.results = clean_results
        self.plot_paths = final_context['displayed_images']

        return None


