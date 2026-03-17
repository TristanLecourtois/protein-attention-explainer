from setuptools import setup, find_packages

setup(
    name="protein-attention-explainer",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "biopython>=1.81",
        "fastapi>=0.104.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "protein-explain=pipeline.run_inference:main",
        ]
    },
)
