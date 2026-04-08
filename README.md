# Hallucination Checker OpenEnv

## Overview

Hallucination Checker OpenEnv is a real-world evaluation environment for Large Language Models (LLMs), designed to measure and analyze hallucination behavior.

Hallucination, defined as generating incorrect or ungrounded information, remains a major obstacle for deploying LLMs in production systems. This environment simulates a fact-checking workflow using live data sources to evaluate grounding, reasoning, and uncertainty handling.

The system dynamically generates tasks using Wikipedia and Arxiv data and evaluates model responses using a multi-metric scoring system.

---

## Key Features

- Dynamic task generation using live Wikipedia and Arxiv data
- Parallel evaluation across three difficulty levels
- Continuous reward scoring between 0.0 and 1.0
- Multi-dimensional hallucination metrics
- LangGraph-based execution workflow
- Built-in adversarial conflict testing

---

## System Architecture

The environment is implemented using a LangGraph workflow that manages task generation, evaluation, and scoring.

### High-Level Flow

1. Environment reset initializes new tasks
2. Tasks are generated in parallel
3. Agent produces responses
4. Evaluation runs in parallel
5. Scores are aggregated into a final reward

---

## Project Structure
├── env.py # Core environment and evaluation logic
├── baseline.py # Baseline agent implementation
├── requirements.txt
├── Dockerfile
└── README.md


![mermaid-diagram](https://cdn-uploads.huggingface.co/production/uploads/64b7d8bf6c169983c992372c/HE3jLIR7HlxNNxpsFysU2.png)

---

## Core Components

### Environment (env.py)

Defines the main evaluation environment.

Key classes:

- HallucinationEnv  
  Provides reset() and step() interface for interaction

- Observation  
  Contains task contexts and questions

- Action  
  Contains agent answers and citations

- Reward  
  Stores scores and metric breakdowns

---

### Baseline Agent (baseline.py)

A simple agent that uses an OpenAI model to answer all tasks.

Behavior:

- Uses task-specific prompts
- Answers each task sequentially
- Submits responses for evaluation
- Prints a detailed report

---

## Task Design

Each evaluation consists of three tasks:

### Task 1: Factual Question Answering (Medium)

- Source: Wikipedia
- Objective: Answer factual questions using only provided context

Evaluation focus:
- Entity correctness
- Numerical accuracy
- Faithfulness to context

---

### Task 2: Research Question Answering (Hard)

- Source: Arxiv abstracts
- Objective: Interpret research findings or limitations

Evaluation focus:
- Grounding in source
- Avoidance of speculation
- Proper uncertainty handling

---

### Task 3: Conflict Resolution (Expert)

- Source: Two conflicting documents
- Objective:
  - Identify contradiction
  - Determine reliable source
  - Provide justified answer

Evaluation focus:
- Contradiction detection
- Reasoning ability
- Source attribution

---

## Evaluation Metrics

Each task is evaluated using the following metrics:

- InfoNCE: Semantic grounding to context
- NLI: Logical consistency with source
- Entity: Accuracy of named entities
- Numbers: Accuracy of numerical data
- Uncertainty: Proper handling of missing information
- Citations: Whether sources are referenced

---

## Reward Calculation

### Task 1 (Factual)
score = 0.4 * Entity + 0.4 * Numbers + 0.2 * NLI

### Task 2 (Arxiv QA)
score = 0.5 * InfoNCE + 0.4 * Uncertainty + 0.1 * Citations

### Task 3 (Conflict Resolution)
score = 0.6 * NLI + 0.2 * InfoNCE + 0.2 * Citations


### Final Reward
final_score = (task1 + task2 + task3) / 3


---

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker (optional)
- OpenAI API key

---

### Install Dependencies
pip install -r requirements.txt


---

### Set API Key

Linux / Mac:
export OPENAI_API_KEY="your-key"

Windows PowerShell:
$env:OPENAI_API_KEY="your-key"


---

## Usage

### Run Baseline
python baseline.py


---

### Example Output
INITIALIZING HALLUCINATION EVALUATOR

Fetching live data...

Agent answering Task 1...
Agent answering Task 2...
Agent answering Task 3...

EVALUATION REPORT CARD

Level 1 Score: 0.78
Level 2 Score: 0.66
Level 3 Score: 0.72

OVERALL REWARD: 0.72



---

## Docker Usage

### Build Image
docker build -t hallucination-env .


### Run Container
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY hallucination-env


---

## Design Highlights

### Dynamic Task Generation

Tasks are generated using:

- WikipediaAPIWrapper
- ArxivAPIWrapper

No static dataset is used.

---

### Parallel Execution

LangGraph enables:

- Parallel task generation
- Parallel evaluation
- Efficient execution

---

### Adversarial Testing

Conflict tasks are created by modifying real data to introduce controlled inconsistencies.

---


## Baseline Performance

- GPT-3.5 Turbo: 0.65 to 0.75
- GPT-4o-mini: approximately 0.70 to 0.80

--

## License

Specify your license here (e.g., MIT License)

---

## Acknowledgements

- LangChain and LangGraph
- OpenAI
- Wikipedia API
- Arxiv API
