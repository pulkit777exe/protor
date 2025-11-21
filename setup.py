from setuptools import setup, find_packages

setup(
    name="protor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "beautifulsoup4",
        "rich",
        "langchain",
        "langchain-community",
        "langchain-ollama",
        "chromadb",
    ],
    entry_points={
        "console_scripts": [
            "protor=protor.cli:cli",
        ],
    },
)