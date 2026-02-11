# INLP-Assignment

## Overview

This project processes and analyzes language data using Python scripts. It includes files for language modeling and tokenization, as well as sample datasets in JSONL format.

## Assumptions

- Python 3.8+ is installed on your system.
- All required packages are standard Python libraries (no external dependencies).
- Input files (`cc100_en.jsonl`, `cc100_mn.jsonl`) are present in the project root directory.
- Scripts (`language_models.py`, `tokenizers.py`) are executable and located in the project root.
- The experiment result included in the report is on the 200k rows from the whole dataset due to computational constraints.

## Execution Commands

To run the language modeling script:
```zsh
python3 language_models.py
```

To run the tokenization script:
```zsh
python3 tokenizers.py
```

## Files

- `language_models.py`: Include language modelling codes with and without smoothing.
- `tokenizers.py`: All the tokenization codes.
- `README.md`: Includes all the assumption and execution command to run this project.
