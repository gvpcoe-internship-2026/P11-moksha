# P6 — ResumeMatch-AI — Semantic Resume ↔ JD Matcher with Explanations

**Student:** MANUKONDA MOKSHAGNA SAI
**Mentor:** Dr. Kanthi Kiran Sirra
**Duration:** 25 May – 19 July 2026

## Problem
Score resume-to-JD fit using embeddings, generate gap explanations via an LLM, and suggest concrete 
resume edits with before/after diffs.

## Approach
[To be filled in by Week 2]

## Tech Stack
sentence-transformers, Chroma, Gemma 3, Streamlit 

## Status
- [ ] Week 1: Foundation
- [ ] Week 2: Literature + baseline
- [ ] Week 3: Initial experiments
- [ ] Week 4: Working prototype (CIS evaluation)
- [ ] Weeks 5–7: Refinement
- [ ] Week 8: Final deliverable

## Demo
[Link here once deployed]



# Sentence Similarity using Sentence Transformers

A simple Python project that demonstrates how to generate sentence embeddings using the **Sentence Transformers** library and calculate the **cosine similarity** between two text strings.

## Objective

The objective of this project is to compare the semantic similarity between two sentences using a pre-trained transformer model.

Example sentences:

- "software engineer with Python experience"
- "looking for Python developer"

The program converts both sentences into vector embeddings and computes their cosine similarity score.

---

## Features

- Python virtual environment setup
- Uses the `sentence-transformers` library
- Generates sentence embeddings
- Calculates cosine similarity
- Simple and beginner-friendly implementation

---

## Project Structure

```
Sentence-Similarity/
│
├── .venv/
├── similarity.py
└── README.md
```

---

## Requirements

- Python 3.10 or later
- sentence-transformers

---

## Installation

### 1. Clone or create the project folder

```bash
mkdir Sentence-Similarity
cd Sentence-Similarity
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate
```

---

## Install Dependencies

```bash
pip install sentence-transformers
```

Verify installation:

```bash
pip show sentence-transformers
```

---

## Code

Create a file named `similarity.py` and add the following code:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load the pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Input sentences
sentence1 = "software engineer with Python experience"
sentence2 = "looking for Python developer"

# Generate embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

# Compute cosine similarity
similarity = cos_sim(embedding1, embedding2)

print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print(f"Cosine Similarity: {similarity.item():.4f}")
```

---

## Run the Project

Execute the following command:

```bash
python similarity.py
```

---

## Sample Output

```
Sentence 1: software engineer with Python experience
Sentence 2: looking for Python developer

Cosine Similarity: 0.72
```

> **Note:** The similarity score may vary slightly depending on the installed model version.

---

## How It Works

1. Load a pre-trained Sentence Transformer model.
2. Convert each sentence into a numerical embedding.
3. Calculate the cosine similarity between the two embeddings.
4. Display the similarity score.

---

## Model Used

- **all-MiniLM-L6-v2**

This lightweight transformer model produces high-quality sentence embeddings and is commonly used for semantic similarity tasks.

---

## Technologies Used

- Python
- Sentence Transformers
- PyTorch
- Hugging Face Transformers

---

## Future Improvements

- Compare multiple sentences.
- Read sentences from a file.
- Build a Streamlit web interface.
- Compare resumes with job descriptions.
- Integrate with semantic search applications.

---

## Author

Developed as a practice project for learning **Sentence Transformers**, **sentence embeddings**, and **semantic similarity** using Python.