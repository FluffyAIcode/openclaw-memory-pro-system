from setuptools import setup, find_packages

setup(
    name="openclaw-memory",
    version="3.0.0",
    description="OpenClaw memory systems: Memora (RAG) + Chronos (CL) + MSA (Sparse Attention) + Second Brain + Memory Server",
    packages=find_packages(),
    install_requires=[
        "rich>=13.7.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0.0",
        "python-dateutil>=2.9.0",
        "requests>=2.31.0",
        "numpy>=1.26.0",
        "networkx>=3.0",
    ],
    extras_require={
        "chronos": [
            "torch>=2.0.0",
        ],
        "embeddings": [
            "sentence-transformers>=3.0.0",
            "einops>=0.8.0",
        ],
        "vectordb": [
            "lancedb>=0.10.0",
        ],
        "web": [
            "streamlit>=1.32.0",
            "streamlit-agraph>=0.0.45",
            "plotly>=5.18.0",
        ],
        "full": [
            "torch>=2.0.0",
            "sentence-transformers>=3.0.0",
            "einops>=0.8.0",
            "lancedb>=0.10.0",
            "streamlit>=1.32.0",
            "streamlit-agraph>=0.0.45",
            "plotly>=5.18.0",
        ],
    },
    py_modules=[
        "memory_hub",
        "memory_hub_cli",
        "memory_cli",
        "memory_server",
        "shared_embedder",
        "llm_client",
    ],
    entry_points={
        "console_scripts": [
            "memora = memora.cli:main",
            "chronos = chronos.cli:main",
            "msa = msa.cli:main",
            "memory-hub = memory_hub_cli:main",
            "memory-cli = memory_cli:main",
            "second-brain = second_brain.cli:main",
        ],
    },
    python_requires=">=3.9",
    package_data={
        "memora": ["config.yaml"],
        "chronos": ["config.yaml"],
        "msa": ["config.yaml"],
        "second_brain": ["config.yaml"],
    },
)
