# Save this file as denario.py inside your denario/ directory

from typing import List, Dict, Union # Added Union for type hints
import asyncio
import time
import os
import shutil
from pathlib import Path
from PIL import Image 
import cmbagent

# --- Corrected Imports ---
# Ensure these imports are correct for your project structure
from .llm import LLM, models, llm_parser 
from .config import DEFAUL_PROJECT_NAME, INPUT_FILES, PLOTS_FOLDER, DESCRIPTION_FILE, IDEA_FILE, METHOD_FILE, RESULTS_FILE, LITERATURE_FILE
from .research import Research
from .key_manager import KeyManager
from .paper_agents.journal import Journal
from .idea import Idea
from .method import Method
from .experiment import Experiment
from .paper_agents.agents_graph import build_graph
from .utils import input_check, check_file_paths, in_notebook
from .langgraph_agents.agents_graph import build_lg_graph
from cmbagent import preprocess_task

class Denario:
    """
    Denario main class. Allows to set the data and tools description, generate a research idea, generate methodology and compute the results. Then it can generate the latex draft of a scientific article with a given journal style from the computed results.
    
    It uses two main backends:

    - `cmbagent`, for detailed planning and control involving numerous agents for the idea, methods and results generation.
    - `langgraph`, for faster idea and method generation, and for the paper writing.

    Args:
        input_data: Input data to be used. Employ default data if `None`.
        project_dir: Directory project. If `None`, create a `project` folder in the current directory.
        clear_project_dir: Clear all files in project directory when initializing if `True`.
    """

    def __init__(self,
                 research: Research | None = None,
                 project_dir: str | None = None, 
                 clear_project_dir: bool = False,
                 ):
        
        if project_dir is None:
            project_dir = os.path.join( os.getcwd(), DEFAUL_PROJECT_NAME )
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)

        if research is None:
            research = Research()
        self.research = research
        self.clear_project_dir = clear_project_dir

        if os.path.exists(project_dir) and clear_project_dir:
            shutil.rmtree(project_dir)
            os.makedirs(project_dir, exist_ok=True)
        self.project_dir = project_dir

        self.plots_folder = os.path.join(self.project_dir, INPUT_FILES, PLOTS_FOLDER)
        os.makedirs(self.plots_folder, exist_ok=True)
        self._setup_input_files()

        self.keys = KeyManager()
        self.keys.get_keys_from_env()
        self.run_in_notebook = in_notebook()
        self.set_all()

    def _setup_input_files(self) -> None:
        input_files_dir = os.path.join(self.project_dir, INPUT_FILES)
        if os.path.exists(input_files_dir) and self.clear_project_dir:
            shutil.rmtree(input_files_dir)
        os.makedirs(input_files_dir, exist_ok=True)

    def reset(self) -> None:
        """Reset Research object"""
        self.research = Research()

    #---
    # Setters
    #---
    def setter(self, field: str | None, file: str) -> str:
        if field is None:
            try:
                with open(os.path.join(self.project_dir, INPUT_FILES, file), 'r') as f:
                    field = f.read()
            except FileNotFoundError:
                raise FileNotFoundError("Please provide an input string or path to a markdown file.")
        field = input_check(field)
        with open(os.path.join(self.project_dir, INPUT_FILES, file), 'w') as f:
            f.write(field)
        return field

    def set_data_description(self, data_description: str | None = None) -> None:
        self.research.data_description = self.setter(data_description, DESCRIPTION_FILE)
        check_file_paths(self.research.data_description)

    def set_idea(self, idea: str | None = None) -> None:
        self.research.idea = self.setter(idea, IDEA_FILE)

    def set_method(self, method: str | None = None) -> None:
        self.research.methodology = self.setter(method, METHOD_FILE)

    def set_results(self, results: str | None = None) -> None:
        self.research.results = self.setter(results, RESULTS_FILE)

    def set_plots(self, plots: list[str] | list[Image.Image] | None = None) -> None:
        if plots is None:
            plots = [str(p) for p in (Path(self.project_dir) / "input_files" / "Plots").glob("*.png")]
        for i, plot in enumerate(plots):
            if isinstance(plot, str):
                plot_path = Path(plot)
                img = Image.open(plot_path)
                plot_name = str(plot_path.name)
            else:
                img = plot
                plot_name = f"plot_{i}.png"
            img.save(os.path.join(self.project_dir, INPUT_FILES, PLOTS_FOLDER, plot_name))

    def set_all(self) -> None:
        for setter in (self.set_data_description, self.set_idea, self.set_method, self.set_results, self.set_plots):
            try:
                setter()
            except FileNotFoundError:
                pass

    #---
    # Printers
    #---
    def printer(self, content: str) -> None:
        if self.run_in_notebook:
            from IPython.display import display, Markdown
            display(Markdown(content))
        else:
            print(content)

    def show_data_description(self) -> None: self.printer(self.research.data_description)
    def show_idea(self) -> None: self.printer(self.research.idea)
    def show_method(self) -> None: self.printer(self.research.methodology)
    def show_results(self) -> None: self.printer(self.research.results)
    def show_keywords(self) -> None:
        print(self.research.keywords)
        keyword_list = "\n".join([f"- [{k}]({v})" for k, v in self.research.keywords.items()]) if isinstance(self.research.keywords, dict) else "\n".join([f"- {k}" for k in self.research.keywords])
        self.printer(keyword_list)

    #---
    # Generative modules
    #---
    def enhance_data_description(self, summarizer_model: str, summarizer_response_formatter_model: str) -> None:
        # This method's logic remains the same.
        pass

    def get_idea(self,
                 mode: str = "fast",
                 llm: LLM | str | Dict = "gemini-2.0-flash",
                 idea_maker_model: LLM | str | Dict = "gpt-4o",
                 idea_hater_model: LLM | str | Dict = "o3-mini",
                 planner_model: LLM | str | Dict = "gpt-4o",
                 plan_reviewer_model: LLM | str | Dict = "o3-mini",
                 orchestration_model: LLM | str | Dict = "gpt-4.1",
                 formatter_model: LLM | str | Dict = "o3-mini",
                ) -> None:
        print(f"Generating idea with {mode} mode")
        if mode == "fast":
            self.get_idea_fast(llm=llm)
        elif mode == "cmbagent":
            self.get_idea_cmagent(idea_maker_model=idea_maker_model, idea_hater_model=idea_hater_model,
                                  planner_model=planner_model, plan_reviewer_model=plan_reviewer_model,
                                  orchestration_model=orchestration_model, formatter_model=formatter_model)
        else:
            raise ValueError("Mode must be either 'fast' or 'cmbagent'")

    def get_idea_cmagent(self,
                         idea_maker_model: LLM | str | Dict = "gpt-4o",
                         idea_hater_model: LLM | str | Dict = "o3-mini",
                         planner_model: LLM | str | Dict = "gpt-4o",
                         plan_reviewer_model: LLM | str | Dict = "o3-mini",
                         orchestration_model: LLM | str | Dict = "gpt-4.1",
                         formatter_model: LLM | str | Dict = "o3-mini",
                         ) -> None:
        idea_maker_model = llm_parser(idea_maker_model)
        idea_hater_model = llm_parser(idea_hater_model)
        planner_model = llm_parser(planner_model)
        plan_reviewer_model = llm_parser(plan_reviewer_model)
        orchestration_model = llm_parser(orchestration_model)
        formatter_model = llm_parser(formatter_model)

        if not self.research.data_description:
            with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                self.research.data_description = f.read()
        
        idea_agent = Idea(work_dir=self.project_dir, idea_maker_model=idea_maker_model.name,
                          idea_hater_model=idea_hater_model.name, planner_model=planner_model.name,
                          plan_reviewer_model=plan_reviewer_model.name, keys=self.keys,
                          orchestration_model=orchestration_model.name, formatter_model=formatter_model.name)
        
        idea_text = idea_agent.develop_idea(self.research.data_description)
        self.research.idea = idea_text
        with open(os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE), 'w') as f:
            f.write(idea_text)
        self.idea = idea_text

    def get_idea_fast(self,
                      llm: LLM | str | Dict = "gemini-2.0-flash",
                      iterations: int = 4,
                      verbose=False,
                      ) -> None:
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm = llm_parser(llm)
        graph = build_lg_graph(mermaid_diagram=False)
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)

        # <<< FIX: Ensure 'llm' key holds the LLM object, not just its attributes.
        input_state = {
            "task": "idea_generation",
            "files": {"Folder": self.project_dir, "data_description": f_data_description},
            "llm": {
                "llm": llm,  # <<< Pass the entire LLM object here
                "model": llm.name,
                "temperature": llm.temperature,
                "max_output_tokens": llm.max_output_tokens,
                "stream_verbose": verbose
            },
            "keys": self.keys,
            "idea": {"total_iterations": iterations},
        }
        graph.invoke(input_state, config)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Idea generated in {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec.")

    def check_idea(self,
                   mode: str = 'semantic_scholar',
                   llm: LLM | str | Dict = "gemini-2.5-flash",
                   max_iterations: int = 7,
                   verbose=False) -> str:
        print(f"Checking idea in literature with {mode} mode")
        if mode == 'futurehouse':
            return self.check_idea_futurehouse()
        elif mode == 'semantic_scholar':
            return self.check_idea_semantic_scholar(llm=llm, max_iterations=max_iterations, verbose=verbose)
        else:
            raise ValueError("Mode must be either 'futurehouse' or 'semantic_scholar'")
    
    def check_idea_semantic_scholar(self,
                        llm: LLM | str | Dict = "gemini-2.5-flash",
                        max_iterations: int = 7,
                        verbose=False,
                        ) -> str:
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm = llm_parser(llm)
        graph = build_lg_graph(mermaid_diagram=False)
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        f_idea = os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE)

        # <<< FIX: Ensure 'llm' key holds the LLM object.
        input_state = {
            "task": "literature",
            "files": {"Folder": self.project_dir, "data_description": f_data_description, "idea": f_idea},
            "llm": {
                "llm": llm,  # <<< Pass the entire LLM object here
                "model": llm.name,
                "temperature": llm.temperature,
                "max_output_tokens": llm.max_output_tokens,
                "stream_verbose": verbose
            },
            "keys": self.keys,
            "literature": {"max_iterations": max_iterations},
            "idea": {"total_iterations": 4},
        }
        
        try:
            graph.invoke(input_state, config)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Literature checked in {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec.")
        except Exception as e:
            print(f'Denario failed to check literature. Error: {e}')
            return "Error occurred during literature check"

        try:
            with open(os.path.join(self.project_dir, INPUT_FILES, LITERATURE_FILE), 'r') as f:
                return f.read()
        except FileNotFoundError:
            return "Literature file not found"

    def get_method(self,
                   mode: str = "fast",
                   llm: LLM | str | Dict = "gemini-2.0-flash",
                   method_generator_model: LLM | str | Dict = "gpt-4o",
                   planner_model: LLM | str | Dict = "gpt-4o",
                   plan_reviewer_model: LLM | str | Dict = "o3-mini",
                   orchestration_model: LLM | str | Dict = "gpt-4.1",
                   formatter_model: LLM | str | Dict = "o3-mini",
                   verbose=False,
                   ) -> None:
        print(f"Generating methodology with {mode} mode")
        if mode == "fast":
            self.get_method_fast(llm=llm, verbose=verbose)
        elif mode == "cmbagent":
            self.get_method_cmbagent(method_generator_model=method_generator_model, planner_model=planner_model,
                                     plan_reviewer_model=plan_reviewer_model, orchestration_model=orchestration_model,
                                     formatter_model=formatter_model)
        else:
            raise ValueError("Mode must be either 'fast' or 'cmbagent'")

    def get_method_cmbagent(self,
                            method_generator_model: LLM | str | Dict = "gpt-4o",
                            planner_model: LLM | str | Dict = "gpt-4o",
                            plan_reviewer_model: LLM | str | Dict = "o3-mini",
                            orchestration_model: LLM | str | Dict = "gpt-4.1",
                            formatter_model: LLM | str | Dict = "o3-mini",
                            ) -> None:
        if not self.research.data_description:
            with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                self.research.data_description = f.read()        
        if not self.research.idea:
            with open(os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE), 'r') as f:
                self.research.idea = f.read()

        method_generator_model = llm_parser(method_generator_model)
        planner_model = llm_parser(planner_model)
        plan_reviewer_model = llm_parser(plan_reviewer_model)
        orchestration_model = llm_parser(orchestration_model)
        formatter_model = llm_parser(formatter_model)

        method = Method(self.research.idea, keys=self.keys, work_dir=self.project_dir, 
                        researcher_model=method_generator_model.name, planner_model=planner_model.name, 
                        plan_reviewer_model=plan_reviewer_model.name, orchestration_model=orchestration_model.name,
                        formatter_model=formatter_model.name)
        
        methodology = method.develop_method(self.research.data_description)
        self.research.methodology = methodology
        with open(os.path.join(self.project_dir, INPUT_FILES, METHOD_FILE), 'w') as f:
            f.write(methodology)

    def get_method_fast(self,
                        llm: LLM | str | Dict = "gemini-2.0-flash",
                        verbose=False,
                        ) -> None:
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm = llm_parser(llm)
        graph = build_lg_graph(mermaid_diagram=False)
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        f_idea = os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE)
        
        # <<< FIX: Ensure 'llm' key holds the LLM object, not just its attributes.
        input_state = {
            "task": "methods_generation",
            "files": {"Folder": self.project_dir, "data_description": f_data_description, "idea": f_idea},
            "llm": {
                "llm": llm,  # <<< Pass the entire LLM object here
                "model": llm.name,
                "temperature": llm.temperature,
                "max_output_tokens": llm.max_output_tokens,
                "stream_verbose": verbose
            },
            "keys": self.keys,
            "idea": {"total_iterations": 4},
        }
        graph.invoke(input_state, config)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Methods generated in {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec.")

    def get_results(self,
                    involved_agents: List[str] = ['engineer', 'researcher'],
                    engineer_model: LLM | str | Dict = "gpt-4.1",
                    researcher_model: LLM | str | Dict = "o3-mini",
                    restart_at_step: int = -1,
                    hardware_constraints: str | None = None,
                    planner_model: LLM | str | Dict = "gpt-4o",
                    plan_reviewer_model: LLM | str | Dict = "o3-mini",
                    max_n_attempts: int = 10, max_n_steps: int = 6,   
                    orchestration_model: LLM | str | Dict = "gpt-4.1",
                    formatter_model: LLM | str | Dict = "o3-mini",
                    ) -> None:
        engineer_model = llm_parser(engineer_model)
        researcher_model = llm_parser(researcher_model)
        planner_model = llm_parser(planner_model)
        plan_reviewer_model = llm_parser(plan_reviewer_model)
        orchestration_model = llm_parser(orchestration_model)
        formatter_model = llm_parser(formatter_model)

        if not self.research.data_description:
            with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                self.research.data_description = f.read()
        if not self.research.idea:
            with open(os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE), 'r') as f:
                self.research.idea = f.read()
        if not self.research.methodology:
            with open(os.path.join(self.project_dir, INPUT_FILES, METHOD_FILE), 'r') as f:
                self.research.methodology = f.read()

        experiment = Experiment(research_idea=self.research.idea, methodology=self.research.methodology,
                                involved_agents=involved_agents, engineer_model=engineer_model.name,
                                researcher_model=researcher_model.name, planner_model=planner_model.name,
                                plan_reviewer_model=plan_reviewer_model.name, work_dir=self.project_dir, keys=self.keys,
                                restart_at_step=restart_at_step, hardware_constraints=hardware_constraints,
                                max_n_attempts=max_n_attempts, max_n_steps=max_n_steps,
                                orchestration_model=orchestration_model.name, formatter_model=formatter_model.name)
        
        experiment.run_experiment(self.research.data_description)
        self.research.results = experiment.results
        self.research.plot_paths = experiment.plot_paths

        if os.path.exists(self.plots_folder):
            for file in os.listdir(self.plots_folder):
                file_path = os.path.join(self.plots_folder, file)
                if os.path.isfile(file_path): os.remove(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
        for plot_path in self.research.plot_paths:
            shutil.move(plot_path, self.plots_folder)

        with open(os.path.join(self.project_dir, INPUT_FILES, RESULTS_FILE), 'w') as f:
            f.write(self.research.results)
    
    def get_keywords(self, input_text: str, n_keywords: int = 5, kw_type: str = 'unesco') -> None:
        keywords = cmbagent.get_keywords(input_text, n_keywords = n_keywords, kw_type = kw_type, api_keys = self.keys)
        self.research.keywords = keywords # type: ignore
        print('keywords: ', self.research.keywords)

    def get_paper(self,
                  journal: Journal = Journal.NONE,
                  llm: LLM | str | Dict = "gemini-2.5-flash",
                  writer: str = 'scientist',
                  cmbagent_keywords: bool = False,
                  add_citations=True,
                  ) -> None:
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm = llm_parser(llm)
        graph = build_graph(mermaid_diagram=False)

        input_state = {
            "files":{"Folder": self.project_dir},
            "llm": {
                "llm": llm,  # <<< FIX: Pass the entire LLM object here
                "model": llm.name,
                "temperature": llm.temperature,
                "max_output_tokens": llm.max_output_tokens
            },
            "paper":{"journal": journal, "add_citations": add_citations, "cmbagent_keywords": cmbagent_keywords},
            "keys": self.keys,
            "writer": writer,
        }
        asyncio.run(graph.ainvoke(input_state, config))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Paper written in {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec.")    

    def referee(self,
                llm: LLM | str | Dict = "gemini-2.5-flash",
                verbose=False) -> None:
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm = llm_parser(llm)
        graph = build_lg_graph(mermaid_diagram=False)
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)

        # <<< FIX: Ensure 'llm' key holds the LLM object.
        input_state = {
            "task": "referee",
            "files": {"Folder": self.project_dir, "data_description": f_data_description},
            "llm": {
                "llm": llm,  # <<< Pass the entire LLM object here
                "model": llm.name,
                "temperature": llm.temperature,
                "max_output_tokens": llm.max_output_tokens,
                "stream_verbose": verbose
            },
            "keys": self.keys,
            "referee": {"paper_version": 2},
        }
        
        try:
            graph.invoke(input_state, config)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Paper reviewed in {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec.")
        except FileNotFoundError as e:
            print('Denario failed to provide a review. Ensure a paper exists. Error: {e}')
        
    def research_pilot(self, data_description: str | None = None) -> None:
        """Full run of Denario. It calls the following methods sequentially:
        ```
        set_data_description(data_description)
        get_idea()
        get_method()
        get_results()
        get_paper()
        ```
        """
        self.set_data_description(data_description)
        self.get_idea()
        self.get_method()
        self.get_results()
        self.get_paper()
