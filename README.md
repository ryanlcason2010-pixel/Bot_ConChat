# Framework Assistant

An adaptive AI tool that helps consultants navigate their proprietary framework library through natural conversation.

## Features

- **Semantic Search**: Find relevant frameworks using natural language queries
- **5 Query Patterns**:
  - **Diagnostic**: Describe symptoms and get framework recommendations
  - **Discovery**: Browse and search the framework library
  - **Details**: Get comprehensive information about specific frameworks
  - **Sequencing**: Understand prerequisites and follow-up frameworks
  - **Comparison**: Compare frameworks side-by-side
- **Adaptive Conversations**: Multi-turn flows with session memory
- **Feedback System**: Thumbs up/down ratings to improve recommendations
- **Fast Responses**: Cached embeddings for sub-2-second search results

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Database**: Excel file (frameworks.xlsx)
- **Vector Search**: NumPy/SciPy cosine similarity
- **Session**: Streamlit session state

## Installation

### Prerequisites

- Python 3.9 or higher
- OpenAI API key

### Mac/Linux

```bash
# Clone the repository
git clone <repository-url>
cd framework-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Windows

```powershell
# Clone the repository
git clone <repository-url>
cd framework-assistant

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your OpenAI API key
```

## Configuration

Edit the `.env` file with your settings:

```env
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional (defaults shown)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
SEARCH_TOP_K=5
SEARCH_MIN_SIMILARITY=0.6
```

## Framework Database

Place your `frameworks.xlsx` file in the `data/` directory.

### Required Columns

| Column | Description |
|--------|-------------|
| id | Unique integer identifier |
| name | Framework name |
| type | Category (e.g., "Sales", "Strategy") |
| business_domains | Comma-separated domains |
| problem_symptoms | Problems this framework addresses |
| use_case | When to use this framework |

### Optional Columns

| Column | Description |
|--------|-------------|
| inputs_required | Required data/inputs |
| outputs_artifacts | What the framework produces |
| diagnostic_questions | Pipe-separated diagnostic questions |
| red_flag_indicators | Warning signs to look for |
| levers | Key actions/levers |
| tags | Comma-separated search tags |
| difficulty_level | beginner/intermediate/advanced |
| related_frameworks | Comma-separated related framework IDs |

See `data/README.md` for detailed schema documentation.

## Usage

### Validate Setup

```bash
python setup_check.py
```

### Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Query Examples

**Diagnostic (Problem-based)**
```
"Client has high employee turnover"
"Struggling with low win rates in enterprise sales"
"Team is burned out and productivity is declining"
```

**Discovery (Browse)**
```
"Show me all sales frameworks"
"What frameworks do you have for strategy?"
"List beginner-friendly frameworks"
```

**Details (Specific framework)**
```
"Tell me about SPIN Selling"
"Explain the MEDDIC framework"
"What is OKR?"
```

**Sequencing (Order/prerequisites)**
```
"What should I do before MEDDIC?"
"What comes after customer discovery?"
"Prerequisites for implementing OKRs"
```

**Comparison**
```
"Compare SPIN Selling vs Consultative Selling"
"What's the difference between OKRs and KPIs?"
"MEDDIC vs BANT - which is better?"
```

## Project Structure

```
framework-assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Configuration template
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── setup_check.py        # Setup validation script
│
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── loader.py         # Excel framework loader
│   ├── embedder.py       # Embedding engine with cache
│   ├── search.py         # Semantic search
│   ├── llm.py            # OpenAI API interface
│   ├── intent.py         # Intent detection
│   └── session.py        # Session management
│
├── handlers/             # Query pattern handlers
│   ├── __init__.py
│   ├── diagnostic.py     # Symptom → Diagnostic
│   ├── discovery.py      # Framework browsing
│   ├── details.py        # Framework details
│   ├── sequencing.py     # Framework sequence
│   └── comparison.py     # Framework comparison
│
├── data/                 # Framework database
│   └── README.md         # Schema documentation
│
├── cache/                # Embedding cache
│   └── README.md         # Cache documentation
│
└── logs/                 # Feedback logs
    └── feedback.json     # User feedback data
```

## Troubleshooting

### "Framework file not found"

Ensure `data/frameworks.xlsx` exists with your framework data.

### "OpenAI API error"

- Check your API key in `.env`
- Verify your OpenAI account has available credits
- Check your API rate limits

### "Embeddings taking too long"

First run generates embeddings for all frameworks. Subsequent runs use cached embeddings. To force regeneration, delete `cache/embeddings_cache.pkl`.

### "Search not finding relevant frameworks"

- Lower `SEARCH_MIN_SIMILARITY` in `.env` (e.g., 0.4)
- Increase `SEARCH_TOP_K` for more results
- Ensure framework data has good `searchable_text` (name, domains, symptoms, use_case, tags)

## Cost Estimates

Based on typical usage with 852 frameworks:

| Operation | Model | Est. Cost |
|-----------|-------|-----------|
| Initial embedding | text-embedding-3-small | ~$0.02 |
| Per query embedding | text-embedding-3-small | <$0.001 |
| Per LLM response | gpt-4o-mini | ~$0.001-0.005 |

**Monthly estimate** (moderate usage: 500 queries):
- Embeddings: ~$0.50
- LLM responses: ~$2.50
- **Total: ~$3/month**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
