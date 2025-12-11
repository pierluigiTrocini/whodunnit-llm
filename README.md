# whodunnit-llm — Bachelor Thesis Project

This repository contains the code and materials for my bachelor (BSc) thesis project. The project studies how large language models (LLMs) identify "perpetrators" in CSI-like scenarios and how they retrieve and attribute supporting sources. The work compares different instruction frameworks (for example, OpenAI guidelines vs CO_STAR Framework) and evaluation conditions (complete vs incomplete information).

This thesis project is based on the work of Frermann et al. (2017): "Whodunnit? Crime Drama as a Case for Natural Language Understanding" — see the References section below.

## Project overview

At a high level, this project:
- Prepares a CSI-like corpus (referenced as a submodule) used for experiments.
- Sends prompts to LLMs to identify perpetrators and request/associate sources.
- Performs source retrieval and links retrieved documents to model outputs.
- Validates model outputs against gold annotations and computes evaluation metrics.
- Produces visualizations to compare approaches and data conditions.

This repository contains scripts to run experiments, helpers for parsing "perpetrator" information, source retrieval tooling, and result validation code. Several PNG visualizations with experimental comparisons are included.

## Repository layout (important files/folders)

- `whodunnit-llm/`
  - `llm_test.py` — main script to run experiments with LLMs (send prompts, collect responses).
  - `perpetrators.py` — functions to extract and manage "perpetrator" entities from responses/data.
  - `source_retrieval_data.py` — prepare and manage data for source retrieval.
  - `results_validation.py` — validate outputs vs. ground truth and compute metrics.
  - `utility.py` — shared helper functions.
  - `whodunnit_llm/` — package submodule (project package code).
  - `previous_prompts/`, `test_results/`, `tests/` — folders for prompts, outputs, and tests.
  - PNG files: comparison charts (CO_STAR vs OpenAI guidelines, complete vs incomplete information), `source_type_distribution.png`, etc.

## Dataset / Submodule

The project references an external corpus (e.g., `EdinburghNLP/csi-corpus`) as a git submodule. This corpus contains the case texts used in experiments.

Be sure to initialize and update submodules after cloning so experiment data is available.

## Requirements

- Git
- Python 3.8+ (or a compatible version; check `pyproject.toml`)
- Poetry (recommended) OR virtualenv + pip
- API keys for any external LLM providers used (e.g., OpenAI). Typical environment variables: `OPENAI_API_KEY` (verify exact names used in the scripts).

## Quick setup (recommended: Poetry)

1. Clone the repository:
   ```bash
   git clone https://github.com/pierluigiTrocini/whodunnit-llm.git
   cd whodunnit-llm
   ```

2. Initialize submodules (download dataset and other submodules):
   ```bash
   git submodule update --init --recursive
   ```

3. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

4. (Optional) Enter a Poetry-managed shell:
   ```bash
   poetry shell
   ```

5. Set required environment variables (example for OpenAI):
   ```bash
   export OPENAI_API_KEY="your_api_key_here"        # macOS / Linux
   # Windows PowerShell:
   # $env:OPENAI_API_KEY="your_api_key_here"
   ```

## Alternative setup (virtualenv + pip)

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux / macOS
   .venn\Scripts\activate       # Windows (PowerShell/CMD)
   ```

2. Export dependencies from Poetry (if using Poetry locally) and install:
   ```bash
   poetry export -f requirements.txt --output requirements.txt --without-hashes
   pip install -r requirements.txt
   ```
   Or manually install packages listed in `pyproject.toml`.

## Running experiments

- Run the main experiment script:
  ```bash
  cd whodunnit-llm
  python llm_test.py
  ```

- Validate results:
  ```bash
  python results_validation.py
  ```

Outputs (by default) are stored under `whodunnit-llm/test_results/` or similar folders—check script configuration or arguments for exact paths.

## Results and visualizations

The repository already includes PNG visualizations comparing experimental conditions, e.g.:
- "Perpetrators identified — complete / incomplete information" for CO_STAR and OpenAI guidelines.
- `source_type_distribution.png`.

Generated experiment outputs are typically saved to `test_results/` — inspect that folder after running experiments.

## Contributing

This repository is organized for a thesis project. If you want to propose improvements:
- Open an issue describing the change.
- Submit a pull request with tests and a clear description of the change.

## License

See the `LICENSE` file in the repository root for licensing details.

## References

Primary paper that inspired and provides the dataset/framework for the CSI-like tasks used in this project:

Lea Frermann, Shay B. Cohen, Mirella Lapata (2017). "Whodunnit? Crime Drama as a Case for Natural Language Understanding." CoRR, abs/1710.11601. URL: http://arxiv.org/abs/1710.11601

## Contact

This is a bachelor thesis project. For questions about the thesis, experiments, or data, contact the author as indicated in the thesis frontmatter or repository metadata.
