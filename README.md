# Dementia Ascertainment from Longitudinal EHR Data

This repository contains research code for evaluating four approaches to retrospective
dementia ascertainment from longitudinal electronic health records:

1. ICD-based rules
2. Keyword-filtered LLM classification
3. Retrieval-augmented generation with LLM classification
4. Traditional machine-learning classification

The original study used protected clinical data from Mass General Brigham. Patient-level
EHR data are not included in this repository and cannot be publicly released. Exact
reproduction of the reported results requires authorized access to the study data and
institutionally approved computing resources.

## Overall workflow

The approaches are evaluated as parallel pipelines against chart-reviewed dementia labels:

    Longitudinal EHR data
             |
             +-- ICD codes --------------------------> Rule-based classification
             |
             +-- Keyword sentence filtering --------> LLM classification
             |
             +-- Section splitting and embedding ---> RAG retrieval ---> LLM classification
             |
             +-- RAG-retrieved evidence ------------> ML classification
             |
             +---------------------------------------> Comparison with chart-review labels

The RAG and keyword pipelines use different methods to select evidence from the clinical
record. The RAG-LLM and ML pipelines can use the same retrieved text, allowing retrieval
and classification performance to be examined separately.

## Installation

Create a dedicated Python environment and install the recorded dependencies:

```bash
pip install -r requirements.txt
```

The dependency file is an environment snapshot rather than a minimal package list.

Some workflows additionally require:

- access to a PostgreSQL database;
- access to an Azure OpenAI deployment;
- local embedding models or an Ollama server;
- storage for persisted Chroma vector databases;
- GPU acceleration for selected local models.

Azure credentials are loaded from the following environment variables:

```text
AZURE_ENDPOINT_URL
AZURE_API_KEY
```

Credentials, protected health information, and patient-level output files must not be
committed to the repository.

## Expected data fields

File names and paths in the notebooks reflect the internal research environment and must
be updated before execution. Common fields referenced by the code include:

| Data source | Common fields |
|---|---|
| Clinical notes | `empi`, `notetxt`, `notetype`, `report_date`, `report_description`, `report_number` |
| Reference labels | `empi`, `label` |
| Patient information | `empi`, `dob` |
| Diagnoses and problem lists | `empi`, `diagnosis_date`, diagnosis description or code |
| RAG evidence export | `empi`, `notes`, `label` |
| Keyword evidence table | `empi`, `record_date`, `medical_summary` |

Some development notebooks also use fields such as `rand_ind` to identify development
subsets.

## Repository workflow

### 1. Rule-based baseline

**Entry point:** `rule based baseline/ADRD_rules_performance.ipynb`

This notebook evaluates two ICD-based dementia classification rules against the
chart-reviewed reference labels. It:

1. connects to PostgreSQL;
2. loads the reference labels and previously generated Rule 1 and Rule 2 classifications;
3. constructs confusion matrices;
4. calculates sensitivity, specificity, PPV, NPV, and F1 score;
5. calculates Wilson confidence intervals for proportion-based metrics.

The notebook expects the database tables or views containing the Rule 1 and Rule 2
outputs to already exist. Database credentials and table names must be configured before
execution.

### 2. Keyword-filtered LLM pipeline

Run the two notebooks in this order.

#### Step 2.1: Aggregate keyword-relevant evidence

**Notebook:** `keyword LLM/keyword aggregation.ipynb`

This notebook:

1. loads longitudinal clinical notes;
2. removes excluded note types such as radiology reports;
3. divides notes into sentences using Stanza with fallback tokenization;
4. identifies sentences containing predefined dementia-related keywords;
5. includes neighboring sentences to preserve context;
6. removes repeated sentences within each patient's record;
7. aggregates the selected evidence by patient and date;
8. optionally combines note evidence with diagnoses, medications, problem lists, and age;
9. writes a patient-level evidence table for LLM classification.

The example output in the notebook is:

```text
data/final_patient_data_table.csv
```

Input and output paths should be changed to match the local environment.

#### Step 2.2: Classify aggregated evidence

**Notebook:** `keyword LLM/keyword aggregated llm.ipynb`

This notebook:

1. loads the keyword-filtered patient evidence and reference labels;
2. orders each patient's evidence chronologically;
3. constructs the dementia classification prompt;
4. submits the evidence to GPT-4o through Azure OpenAI or to a configured Ollama model;
5. parses the structured dementia status and explanation;
6. records processing time, predictions, explanations, and reference labels;
7. writes results incrementally to CSV.

### 3. RAG-LLM pipeline

The RAG workflow contains a preparation stage and an inference stage.

#### Step 3.1: Create section-based note chunks

**Entry point:** `RAG LLM/rule based text splitting/test.py`

The sectionizer detects clinical section headings using the lexicons and parsing utilities
in `RAG LLM/rule based text splitting/`. It can write sectionized notes as combined or
individual files.

Example invocation:

```bash
python "RAG LLM/rule based text splitting/test.py" \
  --input_file path/to/notes.txt \
  --output_dir path/to/sectionized_notes \
  --output_file sections.csv \
  --csv_output
```

Supporting files in the same directory implement section detection, tokenization, section
filtering, temporal processing, database operations, and sectionizer evaluation.

#### Step 3.2: Build the vector database

**Notebook:** `RAG LLM/text_splitting_experiment.ipynb`

This is a configuration and index-building notebook. It contains alternative experiments
rather than a single mandatory execution path.

The notebook supports:

- recursive character splitting;
- alternative separator rules;
- semantic chunking;
- rule-based clinical section splitting;
- MPNet embeddings;
- BioClinicalBERT embeddings;
- Azure `text-embedding-3-large` embeddings;
- persisted Chroma vector stores.

To construct an index:

1. configure the note, label, and sectionized-text paths;
2. select one chunking strategy;
3. select one embedding model;
4. run the cells belonging to that configuration;
5. record the resulting Chroma persistence directory.

Unselected experimental branches do not need to be executed.

#### Step 3.3: Retrieve evidence and run classification

**Notebook:** `RAG LLM/rag llm.ipynb`

The main function is:

```python
run_experiment(
    chunk_strategy,
    embedding_name,
    search_method,
    llm_name,
    result_file,
    prompt_num,
    export_context_only=False,
    test_dataset=False,
)
```

The notebook:

1. loads the appropriate persisted Chroma stores;
2. retrieves patient-specific chunks using cosine similarity or maximal marginal relevance;
3. reranks candidate chunks with a cross-encoder;
4. orders the selected evidence chronologically;
5. optionally adds structured diagnoses and problem-list information;
6. constructs the selected dementia classification prompt;
7. runs GPT-4o through Azure OpenAI or a local Ollama model;
8. records predictions, explanations, prompts, retrieval settings, and latency;
9. either writes classification results or exports retrieved evidence for ML experiments.

Setting `export_context_only=True` writes patient-level retrieved evidence with columns
compatible with the traditional ML workflow:

```text
empi, notes, label
```

`BioClinicalBERTEmbeddings.py`, located under
`ML baseline/2 feature engineering/`, supplies the custom BioClinicalBERT embedding
wrapper used by selected RAG configurations. Its location must be added to the Python
import path when that embedding option is used.

### 4. Traditional machine-learning baselines

Two ML experiment paths are included.

#### 4.1 TF-IDF models using RAG-retrieved evidence

**Primary entry point:** `ML baseline/more ML models/baseline2.py`

The expected input is a patient-level CSV containing:

```text
empi, label, notes
```

The `notes` column should contain the evidence exported by the RAG pipeline.

The script supports:

- logistic regression;
- support vector machine;
- random forest;
- XGBoost.

Its workflow is:

1. load patient-level retrieved evidence;
2. construct TF-IDF features;
3. perform five-fold cross-validation;
4. evaluate configured hyperparameter combinations;
5. select the best configuration;
6. retrain on the full development dataset;
7. evaluate on the independent test dataset;
8. calculate classification metrics and bootstrap confidence intervals;
9. save models, predictions, features, and precision-recall outputs.

Supporting modules in `more ML models/` provide feature extraction, model definitions,
metrics, plotting, file I/O, and statistical utilities. The included `NCRFpp/` directory
contains an adapted external neural NLP framework and supporting experimental code; it
is separate from the primary TF-IDF execution path.

#### 4.2 Transformer-embedding experiments

Run the following components in order.

##### Data preparation

**Notebook:** `ML baseline/1 preprocessing/merge notes with labels.ipynb`

Filters the note table to patients with reference labels, merges notes with labels, and
creates the patient-level dataset used by the embedding workflow.

##### Patient embedding generation

**Files:**

- `ML baseline/2 feature engineering/BioClinicalBERTEmbeddings.py`
- `ML baseline/2 feature engineering/create bioclinicalbert embedding.ipynb`

The Python module implements a LangChain-compatible BioClinicalBERT embedding wrapper.
The notebook can also use other Hugging Face models. It embeds individual notes and
aggregates them into a patient-level representation using max pooling.

The generated development and test embeddings are stored as pickle files.

##### Classification

**Notebooks:**

- `ML baseline/3 machine learning/logistic regression.ipynb`
- `ML baseline/3 machine learning/xgboost.ipynb`

These notebooks:

1. load the development and test embedding files;
2. perform five-fold hyperparameter selection on the development dataset;
3. retrain the selected model on the full development dataset;
4. apply the fixed model to the independent test dataset;
5. calculate performance metrics and bootstrap confidence intervals;
6. save the trained model, patient predictions, and evaluation results.

## Reproduction notes

- The four approaches are parallel experiments; the entire repository is not intended to
  be executed from top to bottom.
- Configure all paths, database connections, model deployments, and output names before
  running a notebook or script.
- Restart the Jupyter kernel and run the selected notebook from its first required cell
  after configuration.
- Do not reuse development labels when evaluating the independent test dataset.
- Keep the retrieval, model, prompt, and threshold configurations fixed when reproducing
  reported test results.
- Exact reported numerical results cannot be reproduced without the protected source data.
- This code evaluates retrospective dementia ascertainment. It is not a prospective
  early-detection system or a clinical decision-support tool.