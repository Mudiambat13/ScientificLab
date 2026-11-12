# Installation

ScientificLab needs **Python 3.11 or newer**. Its dependencies are NumPy, SciPy,
SymPy, Matplotlib and Pint.

```bash
pip install scientificlab
```

From a source checkout (development):

```bash
git clone https://github.com/Mudiambat13/ScientificLab.git
cd ScientificLab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Check the installation:

```bash
python -c "import scientificlab; print(scientificlab.__version__)"
scientificlab-gui --help
```

To build this documentation locally: `pip install -e ".[docs]" && mkdocs serve`.
