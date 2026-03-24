"""
Ground-truth test dataset for benchmarking the memory system.

Each entry has:
- query: the user question
- expected_keywords: keywords that MUST appear in relevant results
- expected_topics: broader topic categories the result should belong to
- ground_truth_answer: the factual answer based on stored memories
- difficulty: easy / medium / hard
"""

RECALL_DATASET = [
    {
        "id": "R01",
        "query": "What vector database did we choose and why?",
        "expected_keywords": ["JSONL", "vector", "Pinecone", "Milvus"],
        "expected_topics": ["architecture", "storage", "database"],
        "ground_truth_answer": "Chose JSONL file storage because data volume is small (<10k), avoiding cloud costs and complexity of Pinecone/Milvus.",
        "difficulty": "easy",
    },
    {
        "id": "R02",
        "query": "What LLM does OpenClaw use as its primary model?",
        "expected_keywords": ["xAI", "Grok", "LLM"],
        "expected_topics": ["model", "LLM", "configuration"],
        "ground_truth_answer": "xAI Grok is the primary LLM, chosen for reasoning capability.",
        "difficulty": "easy",
    },
    {
        "id": "R03",
        "query": "How does the memory server handle slow LLM requests?",
        "expected_keywords": ["ThreadingMixIn", "async", "timeout", "blocking"],
        "expected_topics": ["server", "engineering", "performance"],
        "ground_truth_answer": "Python http.server is single-threaded; ThreadingMixIn was added. LLM-heavy endpoints use async processing with reasonable timeouts.",
        "difficulty": "medium",
    },
    {
        "id": "R04",
        "query": "What is multi-head attention in Transformers?",
        "expected_keywords": ["attention", "Transformer", "head", "subspace"],
        "expected_topics": ["ML", "NLP", "Transformer"],
        "ground_truth_answer": "Multi-head attention lets the model attend to different representation subspaces at different positions simultaneously.",
        "difficulty": "easy",
    },
    {
        "id": "R05",
        "query": "How does quantum error correction work?",
        "expected_keywords": ["qubit", "surface code", "decoherence", "topological"],
        "expected_topics": ["quantum", "computing", "error correction"],
        "ground_truth_answer": "Surface codes encode logical qubits across many physical qubits in 2D lattice. Topological qubits use Majorana fermions for inherent protection.",
        "difficulty": "medium",
    },
    {
        "id": "R06",
        "query": "What is the difference between EWC and LoRA in Chronos?",
        "expected_keywords": ["EWC", "LoRA", "Chronos", "catastrophic forgetting"],
        "expected_topics": ["continual learning", "fine-tuning", "Chronos"],
        "ground_truth_answer": "Chronos uses EWC to prevent catastrophic forgetting and Dynamic LoRA for adapter allocation.",
        "difficulty": "medium",
    },
    {
        "id": "R07",
        "query": "What embedding model does the system use?",
        "expected_keywords": ["nomic", "embed", "768", "sentence"],
        "expected_topics": ["embedding", "model", "vector"],
        "ground_truth_answer": "nomic-ai/nomic-embed-text-v1.5, 768-dimensional with Matryoshka support.",
        "difficulty": "easy",
    },
    {
        "id": "R08",
        "query": "How does quantum cryptography secure communications?",
        "expected_keywords": ["quantum", "cryptography", "QKD", "photon"],
        "expected_topics": ["quantum", "security", "cryptography"],
        "ground_truth_answer": "Quantum cryptography leverages quantum mechanical properties; any eavesdropping disturbs the quantum state, making interception detectable.",
        "difficulty": "medium",
    },
    {
        "id": "R09",
        "query": "What machine learning paradigms exist?",
        "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
        "expected_topics": ["ML", "deep learning", "paradigms"],
        "ground_truth_answer": "Three broad categories: supervised, unsupervised, and reinforcement learning. Deep learning is a subset using multi-layer neural networks.",
        "difficulty": "easy",
    },
    {
        "id": "R10",
        "query": "What was the Memora integration status with OpenClaw?",
        "expected_keywords": ["Memora", "OpenClaw", "集成", "integrated"],
        "expected_topics": ["integration", "Memora", "OpenClaw"],
        "ground_truth_answer": "OpenClaw has successfully integrated the Memora enhanced memory system.",
        "difficulty": "easy",
    },
]

FAITHFULNESS_DATASET = [
    {
        "id": "F01",
        "query": "What storage backend does our memory system use?",
        "expected_grounded_claims": [
            "JSONL file-based vector store",
            "nomic-embed-text for embeddings",
        ],
        "hallucination_traps": [
            "PostgreSQL",
            "Pinecone cloud",
            "ChromaDB",
            "FAISS",
        ],
    },
    {
        "id": "F02",
        "query": "How does the inspiration collision engine work?",
        "expected_grounded_claims": [
            "7 strategies",
            "adaptive weights",
            "novelty scoring",
        ],
        "hallucination_traps": [
            "GPT-4 based scoring",
            "neural collision detection",
            "reinforcement learning rewards",
        ],
    },
    {
        "id": "F03",
        "query": "What is our approach to continual learning?",
        "expected_grounded_claims": [
            "EWC for catastrophic forgetting prevention",
            "personality profile generation",
            "Nebius fine-tuning skeleton",
        ],
        "hallucination_traps": [
            "online gradient descent",
            "experience replay with prioritized sampling",
            "progressive neural networks",
        ],
    },
    {
        "id": "F04",
        "query": "What quantum computing concepts are in our knowledge base?",
        "expected_grounded_claims": [
            "qubits and superposition",
            "surface codes for error correction",
            "quantum cryptography",
        ],
        "hallucination_traps": [
            "quantum machine learning algorithms",
            "Grover's search implementation",
            "quantum neural networks",
        ],
    },
    {
        "id": "F05",
        "query": "How does the skill evolution pipeline work?",
        "expected_grounded_claims": [
            "2-of-3 scoring rule for proposal",
            "utility tracking with feedback",
            "auto-rewrite on low utility",
        ],
        "hallucination_traps": [
            "neural skill embedding",
            "skill distillation network",
            "multi-agent skill transfer",
        ],
    },
]

AB_TEST_QUERIES = [
    {
        "id": "AB01",
        "query": "What technical decisions have we made about our architecture?",
        "eval_criteria": ["specificity", "accuracy", "personalization"],
    },
    {
        "id": "AB02",
        "query": "Summarize what I know about quantum computing.",
        "eval_criteria": ["completeness", "accuracy", "no_hallucination"],
    },
    {
        "id": "AB03",
        "query": "What engineering lessons have we learned?",
        "eval_criteria": ["specificity", "actionability", "grounding"],
    },
    {
        "id": "AB04",
        "query": "What are the main components of the memory system?",
        "eval_criteria": ["completeness", "accuracy", "structure"],
    },
    {
        "id": "AB05",
        "query": "How should I handle slow API calls in a Python HTTP server?",
        "eval_criteria": ["specificity", "actionability", "grounding"],
    },
]
