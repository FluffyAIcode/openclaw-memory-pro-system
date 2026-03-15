from setuptools import setup, find_packages

setup(
    name="memora",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "rich>=13.7.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0.0",
        "python-dateutil>=2.9.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "embeddings": [
            "sentence-transformers>=3.0.0",
            "numpy>=1.26.0",
        ],
        "vectordb": [
            "lancedb>=0.10.0",
        ],
        "web": [
            "streamlit>=1.32.0",
        ],
        "clipboard": [
            "pyperclip>=1.8.2",
        ],
        "full": [
            "sentence-transformers>=3.0.0",
            "numpy>=1.26.0",
            "lancedb>=0.10.0",
            "streamlit>=1.32.0",
            "pyperclip>=1.8.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "memora = memora.cli:main",
        ],
    },
    python_requires=">=3.9",
    package_data={
        "memora": ["config.yaml"],
    },
)
