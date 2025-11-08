from typing import List
import asyncio
import time
import os
import shutil
from pathlib import Path
from PIL import Image 
import cmbagent
import streamlit as st

from .config import DEFAUL_PROJECT_NAME, INPUT_FILES, PLOTS_FOLDER, DESCRIPTION_FILE, IDEA_FILE, METHOD_FILE, RESULTS_FILE, LITERATURE_FILE
from .research import Research
from .key_manager import KeyManager
from .llm import LLM, models
from .local_llm import update_models_with_local_llms, get_vllm_models, get_ollama_models
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
                 vllm_base_url: str | None = None,
                 ollama_host: str | None = None,
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

        self.llm = models['gpt-4o'] # Default LLM

        self.connect_local_llm(vllm_base_url, ollama_host)

        self.set_all()

    def _setup_input_files(self) -> None:
        input_files_dir = os.path.join(self.project_dir, INPUT_FILES)
        
        # If directory exists and want to clear it, remove it and all its contents
        if os.path.exists(input_files_dir) and self.clear_project_dir:
            shutil.rmtree(input_files_dir)
            
        # Create fresh input_files directory
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
        """
        Set the description of the data and tools to be used by the agents.

        Args:
            data_description: String or path to markdown file including the description of the tools and data. If None, assume that a `data_description.md` is present in `project_dir/input_files`.
        """

        self.research.data_description = self.setter(data_description, DESCRIPTION_FILE)

        check_file_paths(self.research.data_description)

    def set_idea(self, idea: str | None = None) -> None:
        """Manually set an idea, either directly from a string or providing the path of a markdown file with the idea."""

        self.research.idea = self.setter(idea, IDEA_FILE)

    def set_method(self, method: str | None = None) -> None:
        """Manually set methods, either directly from a string or providing the path of a markdown file with the methods."""
        
        self.research.methodology = self.setter(method, METHOD_FILE)

    def set_results(self, results: str | None = None) -> None:
        """Manually set the results, either directly from a string or providing the path of a markdown file with the results."""
        
        self.research.results = self.setter(results, RESULTS_FILE)

    def set_plots(self, plots: list[str] | list[Image.Image] | None = None) -> None:
        """Manually set the plots from their path."""

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

    def set_llm(self, llm: LLM | str):
        """Sets the language model to be used by the agent."""
        self.llm = self._llm_parser(llm)

    def set_all(self) -> None:
        """Set all Research fields if present in the working directory"""

        for setter in (
            self.set_data_description,
            self.set_idea,
            self.set_method,
            self.set_results,
            self.set_plots,
        ):
            try:
                setter()
            except FileNotFoundError:
                pass

    #---
    # Printers
    #---

    def printer(self, content: str) -> None:
        """Method to show the content depending on the execution environment, whether Jupyter notebook or Python script."""

        if self.run_in_notebook:
            from IPython.display import display, Markdown
            display(Markdown(content))
        else:
            print(content)

    def show_data_description(self) -> None:
        """Show the data description set by the `set_data_description` method."""

        self.printer(self.research.data_description)

    def show_idea(self) -> None:
        """Show the provided or generated idea by the `set_idea` or `get_idea` methods."""

        self.printer(self.research.idea)

    def show_method(self) -> None:
        """Show the provided or generated methods by `set_method` or `get_method`."""

        self.printer(self.research.methodology)

    def show_results(self) -> None:
        """Show the obtained results."""

        self.printer(self.research.results)

    def show_keywords(self) -> None:
        """Show the keywords."""

        print(self.research.keywords)

        if isinstance(self.research.keywords, dict):
            # Handle dict format (AAS keywords with URLs)
            keyword_list = "\n".join(
                                [f"- [{keyword}]({self.research.keywords[keyword]})" for keyword in self.research.keywords]
                            )
        else:
            # Handle list format (UNESCO keywords)
            keyword_list = "\n".join([f"- {keyword}" for keyword in self.research.keywords])
        
        self.printer(keyword_list)

    #---
    # Generative modules
    #---

    def enhance_data_description(self,
                                 summarizer_model: str, 
                                 summarizer_response_formatter_model: str) -> None:
        """
        Enhance the data description using the preprocess_task from cmbagent.

        Args:
            summarizer_model: LLM to be used for summarization.
            summarizer_response_formatter_model: LLM to be used for formatting the summarization response.
        """

        # Check if data description exists
        if not hasattr(self.research, 'data_description') or not self.research.data_description:
            # Try to load from file if it exists
            try:
                with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                    self.research.data_description = f.read()
            except FileNotFoundError:
                raise ValueError("No data description found. Please set a data description first before enhancing it.")

        # Get the enhanced text from preprocess_task
        enhanced_text = preprocess_task(self.research.data_description,
                                        work_dir = self.project_dir,
                                        summarizer_model = summarizer_model,
                                        summarizer_response_formatter_model = summarizer_response_formatter_model
                                        )
        
        # Debug: Check if the enhanced text is different from original
        print(f"Original text length: {len(self.research.data_description)}")
        print(f"Enhanced text length: {len(enhanced_text)}")
        print(f"Texts are different: {self.research.data_description != enhanced_text}")
        
        # If the enhanced text is the same as original, try reading from enhanced_input.md
        if self.research.data_description == enhanced_text:
            enhanced_input_path = os.path.join(self.project_dir, "enhanced_input.md")
            if os.path.exists(enhanced_input_path):
                print("Reading enhanced content from enhanced_input.md")
                with open(enhanced_input_path, 'r', encoding='utf-8') as f:
                    enhanced_text = f.read()
                print(f"Enhanced text from file length: {len(enhanced_text)}")
        
        # Update the research object with enhanced text
        self.research.data_description = enhanced_text

        # Create the input_files directory if it doesn't exist
        input_files_dir = os.path.join(self.project_dir, INPUT_FILES)
        if not os.path.exists(input_files_dir):
            os.makedirs(input_files_dir, exist_ok=True)

        # Write the enhanced text to data_description.md
        with open(os.path.join(input_files_dir, DESCRIPTION_FILE), 'w', encoding='utf-8') as f:
            f.write(enhanced_text)

        # set the enhanced text to the research object
        self.research.data_description = enhanced_text
            
        print(f"Enhanced text written to: {os.path.join(input_files_dir, DESCRIPTION_FILE)}")

    def get_idea(self,
                 mode = "fast",
                 llm: LLM | str | None = None,
                 idea_maker_model: LLM | str = models["gpt-4o"],
                 idea_hater_model: LLM | str = models["o3-mini"],
                 planner_model: LLM | str = models["gpt-4o"],
                 plan_reviewer_model: LLM | str = models["o3-mini"],
                 orchestration_model: LLM | str = models["gpt-4.1"],
                 formatter_model: LLM | str = models["o3-mini"],
                ) -> None:
        """Generate an idea making use of the data and tools described in `data_description.md`.

        Args:
            mode: either "fast" or "cmbagent". Fast mode uses langgraph backend and is faster but less reliable. Cmbagent mode uses cmbagent backend and is slower but more reliable.
            llm: the LLM to be used for the fast mode. If None, the LLM set in the Denario object will be used.
            idea_maker_model: the LLM to be used for the idea maker agent.
            idea_hater_model: the LLM to be used for the idea hater agent.
            planner_model: the LLM to be used for the planner agent.
            plan_reviewer_model: the LLM to be used for the plan reviewer agent.
            orchestration_model: the LLM to be used for the orchestration of the agents.
            formatter_model: the LLM to be used for formatting the responses of the agents.
        """

        print(f"Generating idea with {mode} mode")

        llm_to_use = llm if llm is not None else self.llm

        if mode == "fast":
            self.get_idea_fast(llm=llm_to_use)
        elif mode == "cmbagent":
            self.get_idea_cmagent(idea_maker_model=idea_maker_model,
                                  idea_hater_model=idea_hater_model,
                                  planner_model=planner_model,
                                  plan_reviewer_model=plan_reviewer_model,
                                  orchestration_model=orchestration_model,
                                  formatter_model=formatter_model)
        else:
            raise ValueError("Mode must be either 'fast' or 'cmbagent'")

    def get_idea_cmagent(self,
                    idea_maker_model: LLM | str = models["gpt-4o"],
                    idea_hater_model: LLM | str = models["o3-mini"],
                    planner_model: LLM | str = models["gpt-4o"],
                    plan_reviewer_model: LLM | str = models["o3-mini"],
                    orchestration_model: LLM | str = models["gpt-4.1"],
                    formatter_model: LLM | str = models["o3-mini"],
                ) -> None:
        """Generate an idea making use of the data and tools described in `data_description.md` with the cmbagent backend.
        
        Args:
            idea_maker_model: the LLM to be used for the idea maker agent.
            idea_hater_model: the LLM to be used for the idea hater agent.
            planner_model: the LLM to be used for the planner agent.
            plan_reviewer_model: the LLM to be used for the plan reviewer agent.
            orchestration_model: the LLM to be used for the orchestration of the agents.
            formatter_model: the LLM to be used for formatting the responses of the agents.
        """

        # Get LLM instances
        idea_maker_model = self._llm_parser(idea_maker_model)
        idea_hater_model = self._llm_parser(idea_hater_model)
        planner_model = self._llm_parser(planner_model)
        plan_reviewer_model = self._llm_parser(plan_reviewer_model)
        orchestration_model = self._llm_parser(orchestration_model)
        formatter_model = self._llm_parser(formatter_model)
        
        if self.research.data_description == "":
            with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                self.research.data_description = f.read()

        idea = Idea(work_dir = self.project_dir,
                    idea_maker_model = idea_maker_model.name,
                    idea_hater_model = idea_hater_model.name,
                    planner_model = planner_model.name,
                    plan_reviewer_model = plan_reviewer_model.name,
                    keys=self.keys,
                    orchestration_model = orchestration_model.name,
                    formatter_model = formatter_model.name)
        
        idea = idea.develop_idea(self.research.data_description)
        self.research.idea = idea
        # Write idea to file
        idea_path = os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE)
        with open(idea_path, 'w') as f:
            f.write(idea)

        self.idea = idea

    def get_idea_fast(self,
                      llm: LLM | str | None = None,
                      iterations: int = 4,
                      verbose=False,
                      ) -> None:
        """
        Generate an idea using the idea maker - idea hater method.
        
        Args:
            llm: the LLM model to be used
            verbose: whether to stream the LLM response
        """

        # Start timer
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit":100}

        # Get LLM instance
        llm_to_use = llm if llm is not None else self.llm
        llm_to_use = self._llm_parser(llm_to_use)

        # Build graph
        graph = build_lg_graph(mermaid_diagram=False)

        # get name of data description file
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)

        # Initialize the state
        input_state = {
            "task": "idea_generation",
            "files":{"Folder": self.project_dir,
                     "data_description": f_data_description}, #name of project folder
            "llm": {"model": llm_to_use.name,                #name of the LLM model to use
                    "temperature": llm_to_use.temperature,
                    "max_output_tokens": llm_to_use.max_output_tokens,
                    "stream_verbose": verbose,
                    "ollama_host": self.ollama_host,
                    "llm_obj": llm_to_use},
            "keys": self.keys,
            "idea": {"total_iterations": iterations},
        }
        
        # Run the graph
        graph.invoke(input_state, config) # type: ignore
        
        # End timer and report duration in minutes and seconds
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Idea generated in {minutes} min {seconds} sec.")

    def check_idea(self,
                   mode : str = 'semantic_scholar',
                   llm: LLM | str | None = None,
                   max_iterations: int = 7,
                   verbose=False) -> str:
        """
        Use Futurehouse or Semantic Scholar to check the idea against previous literature

        Args:
            mode: either 'futurehouse' or 'semantic_scholar'
            llm: the LLM model to be used
            max_iterations: maximum number of iterations to search for literature
            verbose: whether to stream the LLM response
        """

        print(f"Checking idea in literature with {mode} mode")

        if mode == 'futurehouse':
            return self.check_idea_futurehouse()

        elif mode == 'semantic_scholar':
            llm_to_use = llm if llm is not None else self.llm
            return self.check_idea_semantic_scholar(llm=llm_to_use, max_iterations=max_iterations, verbose=verbose)
        
        else:
            raise ValueError("Mode must be either 'futurehouse' or 'semantic_scholar'")
    
    def check_idea_futurehouse(self) -> str:
        """
        Check with the literature if an idea is original or not.
        """

        from futurehouse_client import FutureHouseClient, JobNames
        from futurehouse_client.models import (
            TaskRequest,
        )
        import os
        fhkey = os.getenv("FUTURE_HOUSE_API_KEY")

        fh_client = FutureHouseClient(
            api_key=fhkey,
        )

        check_idea_prompt = rf"""
        Has anyone worked on or explored the following idea?

        {self.research.idea}
        
        <DESIRED_RESPONSE_FORMAT>
        Answer: <yes or no>

        Related previous work: <describe previous literature on the topic>
        </DESIRED_RESPONSE_FORMAT>
        """
        task_data = TaskRequest(name=JobNames.from_string("owl"),
                                query=check_idea_prompt)
        
        task_response = fh_client.run_tasks_until_done(task_data)

        answer = task_response[0].formatted_answer # type: ignore

        ## process the answer to remove everything above </DESIRED_RESPONSE_FORMAT> 
        answer = answer.split("</DESIRED_RESPONSE_FORMAT>")[1]

        # prepend " Has anyone worked on or explored the following idea?" to the answer
        answer = "Has anyone worked on or explored the following idea?\n" + answer

        ## save the response into {INPUT_FILES}/{LITERATURE_FILE}
        with open(os.path.join(self.project_dir, INPUT_FILES, LITERATURE_FILE), 'w') as f:
            f.write(answer)

        return answer

    def check_idea_semantic_scholar(self,
                        llm: LLM | str | None = None,
                        max_iterations: int = 7,
                        verbose=False,
                        ) -> str:
        """
        Check with the literature if an idea is original or not.

        Args:
           llm: the LLM model to be used
           max_iterations: maximum number of iterations to check the idea
           verbose: whether to stream the LLM response 
        """

        # Start timer
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit":100}

        # Get LLM instance
        llm_to_use = llm if llm is not None else self.llm
        llm_to_use = self._llm_parser(llm_to_use)

        # Build graph
        graph = build_lg_graph(mermaid_diagram=False)

        # get name of data description and idea files
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        f_idea             = os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE)

        # Initialize the state
        input_state = {
            "task": "literature",
            "files":{"Folder": self.project_dir, #name of project folder
                     "data_description": f_data_description,
                     "idea": f_idea}, 
            "llm": {"model": llm_to_use.name,                #name of the LLM model to use
                    "temperature": llm_to_use.temperature,
                    "max_output_tokens": llm_to_use.max_output_tokens,
                    "stream_verbose": verbose,
                    "ollama_host": self.ollama_host},
            "keys": self.keys,
            "literature": {"max_iterations": max_iterations},
            "idea": {"total_iterations": 4},
        }
        
        # Run the graph
        try:
            graph.invoke(input_state, config) # type: ignore
            
            # End timer and report duration in minutes and seconds
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print(f"Literature checked in {minutes} min {seconds} sec.")
            
        except Exception as e:
            print('Denario failed to check literature')
            print(f'Error: {e}')
            return "Error occurred during literature check"

        # Read and return the generated literature content
        try:
            literature_file = os.path.join(self.project_dir, INPUT_FILES, LITERATURE_FILE)
            with open(literature_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return "Literature file not found"
        
    def get_method(self,
                   mode = "fast",
                   llm: LLM | str | None = None,
                   method_generator_model: LLM | str = models["gpt-4o"],
                   planner_model: LLM | str = models["gpt-4o"],
                   plan_reviewer_model: LLM | str = models["o3-mini"],
                   orchestration_model: LLM | str = models["gpt-4.1"],
                   formatter_model: LLM | str = models["o3-mini"],
                   verbose = False,
                   ) -> None:
        """
        Generate the methods to be employed making use of the data and tools described in `data_description.md` and the idea in `idea.md`.
        
        Args:
            mode: either "fast" or "cmbagent". Fast mode uses langgraph backend and is faster but less reliable. Cmbagent mode uses cmbagent backend and is slower but more reliable.
            llm: the LLM to be used for the fast mode. If None, the LLM set in the Denario object will be used.
            method_generator_model: (researcher) the LLM model to be used for the researcher agent.
            planner_model: the LLM model to be used for the planner agent.
            plan_reviewer_model: the LLM model to be used for the plan reviewer agent.
            orchestration_model: the LLM model to be used for the orchestration of the agents.
            formatter_model: the LLM model to be used for formatting the responses of the agents.
        """

        print(f"Generating methodology with {mode} mode")

        llm_to_use = llm if llm is not None else self.llm

        if mode == "fast":
            self.get_method_fast(llm=llm_to_use, verbose=verbose)
        elif mode == "cmbagent":
            self.get_method_cmbagent(method_generator_model=method_generator_model,
                                     planner_model=planner_model,
                                     plan_reviewer_model=plan_reviewer_model,
                                     orchestration_model=orchestration_model,
                                     formatter_model=formatter_model)
        else:
            raise ValueError("Mode must be either 'fast' or 'cmbagent'")

    def get_method_cmbagent(self,
                            method_generator_model: LLM | str = models["gpt-4o"],
                            planner_model: LLM | str = models["gpt-4o"],
                            plan_reviewer_model: LLM | str = models["o3-mini"],
                            orchestration_model: LLM | str = models["gpt-4.1"],
                            formatter_model: LLM | str = models["o3-mini"],
                            ) -> None:
        """
        Generate the methods to be employed making use of the data and tools described in `data_description.md` and the idea in `idea.md`.
        
        Args:
            method_generator_model: (researcher) the LLM model to be used for the researcher agent.
            planner_model: the LLM model to be used for the planner agent.
            plan_reviewer_model: the LLM model to be used for the plan reviewer agent.
            orchestration_model: the LLM model to be used for the orchestration of the agents.
            formatter_model: the LLM model to be used for formatting the responses of the agents.
        """

        if self.research.data_description == "":
            with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                self.research.data_description = f.read()        

        if self.research.idea == "":
            with open(os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE), 'r') as f:
                self.research.idea = f.read()

        method_generator_model = self._llm_parser(method_generator_model)
        planner_model = self._llm_parser(planner_model)
        plan_reviewer_model = self._llm_parser(plan_reviewer_model)
        orchestration_model = self._llm_parser(orchestration_model)
        formatter_model = self._llm_parser(formatter_model)

        method = Method(self.research.idea, keys=self.keys,  
                        work_dir = self.project_dir, 
                        researcher_model=method_generator_model.name, 
                        planner_model=planner_model.name, 
                        plan_reviewer_model=plan_reviewer_model.name,
                        orchestration_model = orchestration_model.name,
                        formatter_model = formatter_model.name)
        
        methododology = method.develop_method(self.research.data_description)
        self.research.methodology = methododology

        # Write idea to file
        method_path = os.path.join(self.project_dir, INPUT_FILES, METHOD_FILE)
        with open(method_path, 'w') as f:
            f.write(methododology)

    def get_method_fast(self,
                        llm: LLM | str | None = None,
                        verbose=False,
                        ) -> None:
        """
        Generate the methods to be employed making use of the data and tools described in `data_description.md` and the idea in `idea.md`. Faster version get_method.
        
        Args:
           llm: the LLM model to be used
           verbose: whether to stream the LLM response
        """

        # Start timer
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit":100}

        # Get LLM instance
        llm_to_use = llm if llm is not None else self.llm
        llm_to_use = self._llm_parser(llm_to_use)

        # Build graph
        graph = build_lg_graph(mermaid_diagram=False)

        # get name of data description file and idea
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        f_idea = os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE)
        
        # Initialize the state
        input_state = {
            "task": "methods_generation",
            "files":{"Folder": self.project_dir,              #name of project folder
                     "data_description": f_data_description,
                     "idea": f_idea}, 
            "llm": {"model": llm_to_use.name,                #name of the LLM model to use
                    "temperature": llm_to_use.temperature,
                    "max_output_tokens": llm_to_use.max_output_tokens,
                    "stream_verbose": verbose,
                    "ollama_host": self.ollama_host},
            "keys": self.keys,
            "idea": {"total_iterations": 4},
        }
        
        # Run the graph
        graph.invoke(input_state, config) # type: ignore
        
        # End timer and report duration in minutes and seconds
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Methods generated in {minutes} min {seconds} sec.")  

    def get_results(self,
                    involved_agents: List[str] = ['engineer', 'researcher'],
                    engineer_model: LLM | str = models["gpt-4.1"],
                    researcher_model: LLM | str = models["o3-mini"],
                    restart_at_step: int = -1,
                    hardware_constraints: str | None = None,
                    planner_model: LLM | str = models["gpt-4o"],
                    plan_reviewer_model: LLM | str = models["o3-mini"],
                    max_n_attempts: int = 10,
                    max_n_steps: int = 6,   
                    orchestration_model: LLM | str = models["gpt-4.1"],
                    formatter_model: LLM | str = models["o3-mini"],
                    ) -> None:
        """
        Compute the results making use of the methods, idea and data description.

        Args:
            involved_agents: List of agents employed to compute the results.
            engineer_model: the LLM model to be used for the engineer agent.
            researcher_model: the LLM model to be used for the researcher agent.
            restart_at_step: the step to restart the experiment.
            hardware_constraints: the hardware constraints to be used for the experiment.
            planner_model: the LLM model to be used for the planner agent.
            plan_reviewer_model: the LLM model to be used for the plan reviewer agent.
            orchestration_model: the LLM model to be used for the orchestration of the agents.
            formatter_model: the LLM model to be used for the formatting of the responses of the agents.
            max_n_attempts: the maximum number of attempts to execute code within one step if the code execution fails.
            max_n_steps: the maximum number of steps in the workflow.
        """

        # Get LLM instances
        engineer_model = self._llm_parser(engineer_model)
        researcher_model = self._llm_parser(researcher_model)
        planner_model = self._llm_parser(planner_model)
        plan_reviewer_model = self._llm_parser(plan_reviewer_model)
        orchestration_model = self._llm_parser(orchestration_model)
        formatter_model = self._llm_parser(formatter_model)

        if self.research.data_description == "":
            with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                self.research.data_description = f.read()

        if self.research.idea == "":
            with open(os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE), 'r') as f:
                self.research.idea = f.read()

        if self.research.methodology == "":
            with open(os.path.join(self.project_dir, INPUT_FILES, METHOD_FILE), 'r') as f:
                self.research.methodology = f.read()

        experiment = Experiment(research_idea=self.research.idea,
                                methodology=self.research.methodology,
                                involved_agents=involved_agents,
                                engineer_model=engineer_model.name,
                                researcher_model=researcher_model.name,
                                planner_model=planner_model.name,
                                plan_reviewer_model=plan_reviewer_model.name,
                                work_dir = self.project_dir,
                                keys=self.keys,
                                restart_at_step = restart_at_step,
                                hardware_constraints = hardware_constraints,
                                max_n_attempts=max_n_attempts,
                                max_n_steps=max_n_steps,
                                orchestration_model = orchestration_model.name,
                                formatter_model = formatter_model.name)
        
        experiment.run_experiment(self.research.data_description)
        self.research.results = experiment.results
        self.research.plot_paths = experiment.plot_paths

        # move plots to the plots folder in input_files/plots 
        ## Clearing the folder
        if os.path.exists(self.plots_folder):
            for file in os.listdir(self.plots_folder):
                file_path = os.path.join(self.plots_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        for plot_path in self.research.plot_paths:
            shutil.move(plot_path, self.plots_folder)

        # Write results to file
        results_path = os.path.join(self.project_dir, INPUT_FILES, RESULTS_FILE)
        with open(results_path, 'w') as f:
            f.write(self.research.results)
    
    def get_keywords(self, input_text: str, n_keywords: int = 5, kw_type: str = 'unesco') -> None:
        """
        Get keywords from input text using cmbagent.

        Args:
            input_text (str): Text to extract keywords from
            n_keywords (int, optional): Number of keywords to extract. Defaults to 5.
            kw_type (str, optional): Type of keywords to extract. Defaults to 'unesco'.

        Returns:
            dict: Dictionary mapping keywords to their URLs
        """
        
        keywords = cmbagent.get_keywords(input_text, n_keywords = n_keywords, kw_type = kw_type, api_keys = self.keys)
        self.research.keywords = keywords # type: ignore
        print('keywords: ', self.research.keywords)

    def get_paper(self,
                  journal: Journal = Journal.NONE,
                  llm: LLM | str | None = None,
                  writer: str = 'scientist',
                  cmbagent_keywords: bool = False,
                  add_citations=True,
                  ) -> None:
        """
        Generate a full paper based on the files in input_files:

            - idea.md
            - methods.md
            - results.md
            - plots

        Different journals considered

            - NONE = None : No journal, use standard latex presets with unsrt for bibliography style.
            - AAS  = "AAS" : American Astronomical Society journals, including the Astrophysical Journal.
            - APS = "APS" : Physical Review Journals from the American Physical Society, including Physical Review Letters, PRA, etc.
            - ICML = "ICML" : ICML - International Conference on Machine Learning.
            - JHEP = "JHEP" : Journal of High Energy Physics, including JHEP, JCAP, etc.
            - NeurIPS = "NeurIPS" : NeurIPS - Conference on Neural Information Processing Systems.
            - PASJ = "PASJ" : Publications of the Astronomical Society of Japan.

        Args:
            journal: Journal style. The paper generation will use the presets of the journal considered for the latex writing. Default is no journal (no specific presets).
            llm: The LLM model to be used to write the paper.
            writer: set the style and tone to write. E.g. astrophysicist, biologist, chemist
            cmbagent_keywords: whether to use CMBAgent to select the keywords
            add_citations: whether to add citations to the paper or not
        """
        
        # Start timer
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit":100}

        # Get LLM instance
        llm_to_use = llm if llm is not None else self.llm
        llm_to_use = self._llm_parser(llm_to_use)

        # Build graph
        graph = build_graph(mermaid_diagram=False)

        # Initialize the state
        input_state = {
            "files":{"Folder": self.project_dir}, #name of project folder
            "llm": {"model": llm_to_use.name,  #name of the LLM model to use
                    "temperature": llm_to_use.temperature,
                    "max_output_tokens": llm_to_use.max_output_tokens},
            "paper":{"journal": journal, "add_citations": add_citations,
                     "cmbagent_keywords": cmbagent_keywords},
            "keys": self.keys,
            "writer": writer,
        }

        # Run the graph
        asyncio.run(graph.ainvoke(input_state, config)) # type: ignore
        
        # End timer and report duration in minutes and seconds
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Paper written in {minutes} min {seconds} sec.")    

    def referee(self,
                llm: LLM | str | None = None,
                verbose=False) -> None:
        """
        Review a paper, producing a report providing feedback on the quality of the articled and aspects to be improved.

        Args:
           llm: the LLM model to be used
           verbose: whether to stream the LLM response 
        """

        # Start timer
        start_time = time.time()
        config = {"configurable": {"thread_id": "1"}, "recursion_limit":100}

        # Get LLM instance
        llm_to_use = llm if llm is not None else self.llm
        llm_to_use = self._llm_parser(llm_to_use)

        # Build graph
        graph = build_lg_graph(mermaid_diagram=False)

        # get name of data description file and referee
        f_data_description = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)

        # Initialize the state
        input_state = {
            "task": "referee",
            "files":{"Folder": self.project_dir,  #name of project folder
                     "data_description": f_data_description}, 
            "llm": {"model": llm_to_use.name,                #name of the LLM model to use
                    "temperature": llm_to_use.temperature,
                    "max_output_tokens": llm_to_use.max_output_tokens,
                    "stream_verbose": verbose,
                    "ollama_host": self.ollama_host},
            "keys": self.keys,
            "referee": {"paper_version": 2},
        }
        
        # Run the graph
        try:
            graph.invoke(input_state, config) # type: ignore
            
            # End timer and report duration in minutes and seconds
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print(f"Paper reviewed in {minutes} min {seconds} sec.")
            
        except FileNotFoundError as e:
            print('Denario failed to provide a review for the paper. Ensure that a paper in the `paper` folder ex')
            print(f'Error: {e}')
        
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

    def _llm_parser(self, llm: LLM | str, vllm_base_url: str | None = None, ollama_host: str | None = None) -> LLM:
        """Get the LLM instance from a string."""
        if isinstance(llm, str):
            try:
                llm = models[llm]
            except KeyError:
                raise KeyError(f"LLM '{llm}' not available. Please select from: {list(models.keys())}")

        if llm.model_type == "local":
            if llm.client == "vllm":
                try:
                    from openai import OpenAI
                except ImportError:
                    raise ImportError("The `openai` package is required for vLLM support. Please install it with `pip install openai`.")
                base_url = vllm_base_url if vllm_base_url else self.vllm_base_url if self.vllm_base_url else "http://localhost:8000/v1"
                llm._client = OpenAI(base_url=base_url)
            elif llm.client == "ollama":
                self.ollama_host = ollama_host if ollama_host else self.ollama_host if self.ollama_host else "http://localhost:11434"
        return llm

    def connect_local_llm(self, vllm_base_url: str | None = None, ollama_host: str | None = None):
        """
        Connect to local LLMs and update the available models.
        """
        self.vllm_base_url = vllm_base_url
        self.ollama_host = ollama_host
        update_models_with_local_llms(vllm_base_url=self.vllm_base_url, ollama_host=self.ollama_host)

    def get_local_models(self):
        """Get a dictionary of available local models, grouped by client."""
        local_models = {"vllm": [], "ollama": []}
        for model_name, model in models.items():
            if model.model_type == "local":
                if model.client == "vllm":
                    local_models["vllm"].append(model.name)
                elif model.client == "ollama":
                    local_models["ollama"].append(model.name)
        return local_models

    def render_ui(self):
        """Render the Streamlit UI for the Denario application."""
        try:
            st.set_page_config(layout="wide")
        except st.errors.StreamlitAPIException:
            # Most likely already set
            pass

        with st.sidebar:
            st.header("API keys")
            st.write("Input OpenAI, Anthropic, Gemini and Perplexity API keys below.")
            st.write("See [here](https://i-cog.github.io/denario/getting_started/keys/) for more information.")
            with st.expander("Set API keys"):
                self.keys.openai_api_key = st.text_input("OpenAI API Key", value=self.keys.openai_api_key or "", type="password")
                self.keys.anthropic_api_key = st.text_input("Anthropic API Key", value=self.keys.anthropic_api_key or "", type="password")
                self.keys.google_api_key = st.text_input("Google API Key", value=self.keys.google_api_key or "", type="password")
                self.keys.perplexity_api_key = st.text_input("Perplexity API Key", value=self.keys.perplexity_api_key or "", type="password")
                if st.button("Set Keys"):
                    self.keys.set_keys_in_env()
                    st.success("API keys set.")

            st.header("LLM Configuration")
            with st.expander("Set LLM"):
                llm_source = st.radio("Select LLM Source", ("External", "Local"), key="source")

                if llm_source == "External":
                    external_models = {k: v for k, v in models.items() if v.model_type != "local"}
                    model_name = st.selectbox("Select Model", options=list(external_models.keys()), key="model")
                    if model_name:
                        self.set_llm(models[model_name])
                        st.session_state['llm_name'] = model_name

                else: # Local
                    col1, col2 = st.columns(2)
                    with col1:
                        provider = st.selectbox("Select Provider", ("vLLM", "Ollama"), key="provider")
                        ip = st.text_input("IP Address", "localhost", key="ip")
                    with col2:
                        port = st.text_input("Port", "8000" if provider == "vLLM" else "11434", key="port")

                    if st.button("Fetch Models", key="fetch"):
                        if provider == "vLLM":
                            url = f"http://{ip}:{port}/v1"
                            self.vllm_base_url = url
                            st.session_state['local_models'] = get_vllm_models(url)
                        else: # Ollama
                            url = f"http://{ip}:{port}"
                            self.ollama_host = url
                            st.session_state['local_models'] = get_ollama_models(url)

                    if 'local_models' in st.session_state and st.session_state['local_models']:
                        local_model_name = st.selectbox("Select Model", options=st.session_state['local_models'], key="local_model")
                        if local_model_name:
                            self.set_llm(LLM(name=local_model_name,
                                                max_output_tokens=8192,
                                                temperature=0.7,
                                                model_type="local",
                                                client=provider.lower()))
                            st.session_state['llm_name'] = local_model_name
                    elif 'local_models' in st.session_state:
                        st.warning("No models found at the specified address.")

            st.write(f"**Current LLM:** {self.llm.name}")


            st.header("Upload data")
            st.write("Upload the data files")
            uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True, label_visibility="collapsed", key="data_uploader")
            if uploaded_files:
                input_files_dir = os.path.join(self.project_dir, INPUT_FILES)
                os.makedirs(input_files_dir, exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(input_files_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"File '{uploaded_file.name}' uploaded to '{input_files_dir}'.")

            st.header("Download project")
            zip_path = f"{self.project_dir}.zip"
            # Create a zip of the project directory for download
            shutil.make_archive(self.project_dir, 'zip', self.project_dir)
            with open(zip_path, "rb") as fp:
                st.download_button(
                    label="Download all project files",
                    data=fp,
                    file_name=f"{os.path.basename(self.project_dir)}.zip",
                    mime="application/zip"
                )

        st.title("Denario")
        st.write("AI agents to assist the development of a scientific research process.")
        st.write("From developing research ideas, developing methods, computing results and writing or reviewing papers.")
        st.warning("This is a demo deployment of Denario on Hugging Face Spaces. Your session will expire if you close the tab or refresh the page. Recall to download your project files from the sidebar before leaving! ⚠️")

        st.markdown(
            "<span><a href='https://github.com/i-cog/denario' target='_blank'>Project Page</a> | <a href='https://i-cog.github.io/denario/' target='_blank'>Documentation</a> | <a href='https://github.com/i-cog/denario' target='_blank'>Code</a></span>",
            unsafe_allow_html=True
        )

        tab_names = ["Input prompt", "Idea", "Methods", "Analysis", "Paper", "Literature review", "Referee report", "Keywords"]
        tabs = st.tabs(tab_names)

        with tabs[0]: # Input prompt
            st.header("Input prompt")
            st.write("Describe the data and tools to be used in the project. You may also include information about the computing resources required.")

            if 'data_description' not in st.session_state:
                try:
                    with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE), 'r') as f:
                        st.session_state.data_description = f.read()
                except FileNotFoundError:
                    st.session_state.data_description = ""

            data_description = st.text_area("E.g. Analyze the experimental data stored in /path/to/data.csv using sklearn and pandas. This data includes time-series measurements from a particle detector.",
                                            value=st.session_state.data_description, height=200, label_visibility="collapsed")

            if data_description != st.session_state.data_description:
                st.session_state.data_description = data_description
                self.set_data_description(data_description)
                st.rerun()

            st.write("Alternatively, upload a file with the data description in markdown format.")
            uploaded_file = st.file_uploader("Drag and drop file here", type=['md'], label_visibility="collapsed", key="md_uploader")
            if uploaded_file:
                data_description_from_file = uploaded_file.read().decode("utf-8")
                if data_description_from_file != st.session_state.data_description:
                    st.session_state.data_description = data_description_from_file
                    self.set_data_description(data_description_from_file)
                    st.rerun()

            with st.expander("Enhance Data Description Options"):
                st.write("Select models to enhance the data description. The currently selected global LLM will be used by default.")

                all_model_names = list(models.keys())

                try:
                    current_llm_index = all_model_names.index(self.llm.name)
                except ValueError:
                    current_llm_index = all_model_names.index('gpt-4o') if 'gpt-4o' in all_model_names else 0

                summarizer_model_name = st.selectbox(
                    "Summarizer Model",
                    options=all_model_names,
                    index=current_llm_index,
                    key="summarizer_model"
                )
                formatter_model_name = st.selectbox(
                    "Formatter Model",
                    options=all_model_names,
                    index=current_llm_index,
                    key="formatter_model"
                )

                if st.button("Enhance Data Description"):
                    if not self.research.data_description:
                        st.error("Please provide a data description first.")
                    else:
                        with st.spinner("Enhancing data description..."):
                            try:
                                self.enhance_data_description(
                                    summarizer_model=summarizer_model_name,
                                    summarizer_response_formatter_model=formatter_model_name
                                )
                                st.success("Data description enhanced successfully.")
                                st.session_state.data_description = self.research.data_description
                                st.rerun()
                            except Exception as e:
                                st.error(f"An error occurred: {e}")

            st.header("Current data description")
            st.markdown(self.research.data_description or "Data description not set.")

        with tabs[1]:
            st.header("Idea")
            if st.button("Generate Idea"):
                with st.spinner("Generating idea..."):
                    self.get_idea()
                    st.rerun()
            st.markdown(self.research.idea or "No idea generated yet.")

        with tabs[2]:
            st.header("Methods")
            if st.button("Generate Methods"):
                with st.spinner("Generating methods..."):
                    self.get_method()
                    st.rerun()
            st.markdown(self.research.methodology or "No methods generated yet.")

        with tabs[3]:
            st.header("Analysis")
            if st.button("Get Results"):
                with st.spinner("Getting results..."):
                    self.get_results()
                    st.rerun()
            st.markdown(self.research.results or "No results generated yet.")

        with tabs[4]:
            st.header("Paper")
            if st.button("Generate Paper"):
                with st.spinner("Generating paper..."):
                    self.get_paper()
                    st.rerun()
            # Placeholder for paper content

        with tabs[5]:
            st.header("Literature review")
            if st.button("Check Literature"):
                with st.spinner("Checking literature..."):
                    self.check_idea()
                    st.rerun()
            # Placeholder for literature review content

        with tabs[6]:
            st.header("Referee report")
            if st.button("Generate Referee Report"):
                with st.spinner("Generating referee report..."):
                    self.referee()
                    st.rerun()
            # Placeholder for referee report content

        with tabs[7]:
            st.header("Keywords")
            if st.button("Generate Keywords"):
                with st.spinner("Generating keywords..."):
                    self.get_keywords(self.research.idea + "\n" + self.research.methodology + "\n" + self.research.results)
                    st.rerun()
            self.show_keywords()
