# Denario

[![Version](https://img.shields.io/pypi/v/denario.svg)](https://pypi.python.org/pypi/denario) [![Python Version](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](https://www.python.org/downloads/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/denario)](https://pypi.python.org/pypi/denario) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AstroPilot-AI/Denario)
<a href="https://www.youtube.com/@denario-ai" target="_blank">
<img src="https://img.shields.io/badge/YouTube-Subscribe-red?style=flat-square&logo=youtube" alt="Subscribe on YouTube" width="140"/>
</a>

Denario is a multiagent system designed to automatize scientific research. Denario implements AI agents with [AG2](https://ag2.ai/) and [LangGraph](https://www.langchain.com/langgraph), using [cmbagent](https://github.com/CMBAgents/cmbagent) as the research analysis backend.

## Resources

- [🌐 Project page](https://astropilot-ai.github.io/DenarioPaperPage/)

<!-- - [📄 Paper](arxivblabla) -->

- [📖 Documentation](https://denario.readthedocs.io/en/latest/)

- [🖥️ Denario GUI repository](https://github.com/AstroPilot-AI/DenarioApp)

- [🤗 Demo web app for Denario GUI](https://huggingface.co/spaces/astropilot-ai/Denario)

- [📝 End-to-end research papers generated with Denario](https://github.com/AstroPilot-AI/DenarioExamplePapers)

- [🎥 YouTube channel](https://www.youtube.com/@denario-ai)

## Installation

To install denario create a virtual environment and pip install it. We recommend using Python 3.12:

```bash
python -m venv Denario_env
source Denario_env/bin/activate
pip install denario[app] # if this doesn't work do: pip install "denario[app]"
```

Or alternatively install it with [uv](https://docs.astral.sh/uv/), initializing a project and installing it:

```bash
uv init
uv add denario[app]
```

Then, run the gui with:

```
denario run
```

## Get started

Initialize a `Denario` instance and describe the data and tools to be employed.

```python
from denario import Denario

den = Denario(project_dir="project_dir")

prompt = """
Analyze the experimental data stored in data.csv using sklearn and pandas.
This data includes time-series measurements from a particle detector.
"""

den.set_data_description(prompt)
```

Generate a research idea from that data specification.

```python
den.get_idea()
```

Generate the methodology required for working on that idea.

```python
den.get_method()
```

With the methodology setup, perform the required computations and get the plots and results.

```python
den.get_results()
```

Finally, generate a latex article with the results. You can specify the journal style, in this example we choose the [APS (Physical Review Journals)](https://journals.aps.org/) style.

```python
from denario import Journal

den.get_paper(journal=Journal.APS)
```

You can also manually provide any info as a string or markdown file in an intermediate step, using the `set_idea`, `set_method` or `set_results` methods. For instance, for providing a file with the methodology developed by the user:

```python
den.set_method(path_to_the_method_file.md)
```

## DenarioApp

You can run Denario using a GUI through the [DenarioApp](https://github.com/AstroPilot-AI/DenarioApp).

The app is already installed with `pip install "denario[app]"`, otherwise install it with `pip install denario_app` or `uv sync --extra app`.

Then, launch the GUI with

```bash
denario run
```

Test a [deployed demo of the app in HugginFace Spaces](https://huggingface.co/spaces/astropilot-ai/Denario).

## Build from source

### pip

You will need python 3.12 or higher installed. Clone Denario:

```bash
git clone https://github.com/AstroPilot-AI/Denario.git
cd Denario
```

Create and activate a virtual environment

```bash
python3 -m venv Denario_env
source Denario_env/bin/activate
```

And install the project

```bash
pip install -e .
```

### uv

You can also install the project using [uv](https://docs.astral.sh/uv/), just running:

```bash
uv sync
```

which will create the virtual environment and install the dependencies and project. Activate the virtual environment if needed with

```bash
source .venv/bin/activate
```

## Contributing

Pull requests are welcome! Feel free to open an issue for bugs, comments, questions and suggestions.

<!-- ## Citation

If you use this library please link this repository and cite [arXiv:2506.xxxxx](arXiv:x2506.xxxxx). -->

## License

[GNU GENERAL PUBLIC LICENSE (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html)

Denario - Copyright (C) 2025 Pablo Villanueva-Domingo, Francisco Villaescusa-Navarro, Boris Bolliet

## Contact and Enquieries

E-mail: [denario.astropilot.ai@gmail.com](mailto:denario.astropilot.ai@gmail.com)
