from typing import List, Dict # <<< CHANGE: Added Dict for type hinting
import asyncio
import time
import os
import shutil
from pathlib import Path
from PIL import Image 
import cmbagent

# --- Corrected Imports ---
# <<< CHANGE: Consolidated all imports from .llm and removed the duplicate llm_parser from .utils
from .llm import LLM, models, llm_parser 
from .config import DEFAUL_PROJECT_NAME, INPUT_FILES, PLOTS_FOLDER, DESCRIPTION_FILE, IDEA_FILE, METHOD_FILE, RESULTS_FILE, LITERATURE_FILE
from .research import Research
from .key_manager import KeyManager
from .paper_agents.journal import Journal
from .idea import Idea
from .method import Method
from .experiment import Experiment
from .paper_agents.agents_graph import build_graph
from .utils import input_check, check_file_paths, in_notebook # <<< CHANGE: Removed llm_parser from here
from .langgraph_agents.agents_graph import build_lg_graph
from cmbagent import preprocess_task

class Denario:
    """
    Denario main class. Allows to set the data and tools description, generate a research idea, generate methodology and compute the results. The it can generate the latex draft of a scientific article with a given journal style from the computed results.
    
    It uses two main backends:

    - `cmbagent`,  for detailed planning and control involving numerous agents for the idea, methods and results generation.
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
            research = Research()  # Initialize with default values
        self.research = research
        self.clear_project_dir = clear_project_dir

        if os.path.exists(project_dir) and clear_project_dir:
            shutil.rmtree(project_dir)
            os.makedirs(project_dir, exist_ok=True)
        self.project_dir = project_dir

        self.plots_folder = os.path.join(self.project_dir, INPUT_FILES, PLOTS_FOLDER)
        # Ensure the folder exists
        os.makedirs(self.plots_folder, exist_ok=True)

        self._setup_input_files()

        # Get keys from environment if they exist
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
        """Base method for setting the content of idea, method or results."""

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
            if isinstance(plot,str):
                plot_path= Path(plot)
                img = Image.open(plot_path)
                plot_name = str(plot_path.name)
            else:
                img = plot
                plot_name = f"plot_{i}.png"
            img.save( os.path.join(self.project_dir, INPUT_FILES, PLOTS_FOLDER, plot_name) )

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

    def show_data_description(self) -> None:
        self.printer(self.research.data_description)
    def show_idea(self) -> None:
        self.printer(self.research.idea)
    def show_method(self) -> None:
        self.printer(self.research.methodology)
    def show_results(self) -> None:
        self.printer(self.research.results)
    def show_keywords(self) -> None:
        print(self.research.keywords)
        if isinstance(self.research.keywords, dict):
            keyword_list = "\n".join([f"- [{keyword}]({self.research.keywords[keyword]})" for keyword in self.research.keywords])
        else:
            keyword_list = "\n".join([f"- {keyword}" for keyword in self.research.keywords])
        self.printer(keyword_list)

    #---
    # Generative modules
    #---

    def enhance_data_description(self, summarizer_model: str, summarizer_response_formatter_model: str) -> None:
        # This method's logic remains the same.
        pass

    def get_idea(self,
                 mode = "fast",
                 llm: LLM | str | Dict = "gemini-2.0-flash", # <<< CHANGE
                 idea_maker_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                 idea_hater_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                 planner_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                 plan_reviewer_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                 orchestration_model: LLM | str | Dict = "gpt-4.1", # <<< CHANGE
                 formatter_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
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
                    idea_maker_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                    idea_hater_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                    planner_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                    plan_reviewer_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                    orchestration_model: LLM | str | Dict = "gpt-4.1", # <<< CHANGE
                    formatter_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                ) -> None:
        
        # Get LLM instances using the CORRECT parser from .llm
        idea_maker_model = llm_parser(idea_maker_model)
        idea_hater_model = llm_parser(idea_hater_model)
        planner_model = llm_parser(planner_model)
        plan_reviewer_model = llm_parser(plan_reviewer_model)
        orchestration_model = llm_parser(orchestration_model)
        formatter_model = llm_parser(formatter_model)
        
        if self.research.data_description == "":
            with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                self.research.data_description = f.read()

        idea = Idea(work_dir = self.project_dir, idea_maker_model = idea_maker_model.name,
                    idea_hater_model = idea_hater_model.name, planner_model = planner_model.name,
                    plan_reviewer_model = plan_reviewer_model.name, keys=self.keys,
                    orchestration_model = orchestration_model.name, formatter_model = formatter_model.name)
        
        idea = idea.develop_idea(self.research.data_description)
        self.research.idea = idea
        with open(os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE), 'w') as f:
            f.write(idea)
        self.idea = idea

    def get_idea_fast(self,
                      llm: LLM | str | Dict = "gemini-2.0-flash", # <<< CHANGE
                      iterations: int = 4, verbose=False) -> None:
        
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit":100}
        llm = llm_parser(llm) # <<< Use the correct parser
        graph = build_lg_graph(mermaid_diagram=False)
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        input_state = { "task": "idea_generation", "files":{"Folder": self.project_dir, "data_description": f_data_description},
                        "llm": {"model": llm.name, "temperature": llm.temperature, "max_output_tokens": llm.max_output_tokens, "stream_verbose": verbose},
                        "keys": self.keys, "idea": {"total_iterations": iterations}, }
        graph.invoke(input_state, config)
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Idea generated in {minutes} min {seconds} sec.")

    def check_idea(self, mode : str = 'semantic_scholar', llm: LLM | str | Dict = "gemini-2.5-flash", # <<< CHANGE
                   max_iterations: int = 7, verbose=False) -> str:
        
        print(f"Checking idea in literature with {mode} mode")
        if mode == 'futurehouse':
            return self.check_idea_futurehouse()
        elif mode == 'semantic_scholar':
            return self.check_idea_semantic_scholar(llm=llm, max_iterations=max_iterations, verbose=verbose)
        else:
            raise ValueError("Mode must be either 'futurehouse' or 'semantic_scholar'")
    
    # ... other methods like check_idea_futurehouse remain the same ...
    
    def get_method(self,
                   mode = "fast",
                   llm: LLM | str | Dict = "gemini-2.0-flash", # <<< CHANGE
                   method_generator_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                   planner_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                   plan_reviewer_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                   orchestration_model: LLM | str | Dict = "gpt-4.1", # <<< CHANGE
                   formatter_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                   verbose = False,
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
                            method_generator_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                            planner_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                            plan_reviewer_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                            orchestration_model: LLM | str | Dict = "gpt-4.1", # <<< CHANGE
                            formatter_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                            ) -> None:

        if self.research.data_description == "":
            with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                self.research.data_description = f.read()        
        if self.research.idea == "":
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

    def get_results(self,
                    involved_agents: List[str] = ['engineer', 'researcher'],
                    engineer_model: LLM | str | Dict = "gpt-4.1", # <<< CHANGE
                    researcher_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                    restart_at_step: int = -1,
                    hardware_constraints: str | None = None,
                    planner_model: LLM | str | Dict = "gpt-4o", # <<< CHANGE
                    plan_reviewer_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                    max_n_attempts: int = 10, max_n_steps: int = 6,   
                    orchestration_model: LLM | str | Dict = "gpt-4.1", # <<< CHANGE
                    formatter_model: LLM | str | Dict = "o3-mini", # <<< CHANGE
                    ) -> None:

        # Get LLM instances
        engineer_model = llm_parser(engineer_model)
        researcher_model = llm_parser(researcher_model)
        planner_model = llm_parser(planner_model)
        plan_reviewer_model = llm_parser(plan_reviewer_model)
        orchestration_model = llm_parser(orchestration_model)
        formatter_model = llm_parser(formatter_model)

        # ... rest of the function logic ...
    
    # ... other methods like get_keywords ...

    def get_paper(self,
                  journal: Journal = Journal.NONE,
                  llm: LLM | str | Dict = "gemini-2.5-flash", # <<< CHANGE
                  writer: str = 'scientist',
                  cmbagent_keywords: bool = False, add_citations=True) -> None:
        
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit":100}
        llm = llm_parser(llm) # <<< Use the correct parser
        # ... rest of the function logic ...

    def referee(self,
                llm: LLM | str | Dict = "gemini-2.5-flash", # <<< CHANGE
                verbose=False) -> None:
        
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit":100}
        llm = llm_parser(llm) # <<< Use the correct parser
        # ... rest of the function logic ...
        
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
