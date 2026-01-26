from setuptools import setup, find_packages

setup(
    name="global_chess_challenge_baselines",
    version="0.1.0",
    description="Baselines for the Global Chess Challenge 2025",
    packages=find_packages(),
    py_modules=[
        "chess_llm", 
        "chess_evaluation_callback", 
        "run_evaluation", 
        "train_nvidia",
        "train_h100_sft"
    ],
    install_requires=[
        "chess",
        "torch",
        "transformers",
        "datasets",
        "wandb",
        "jinja2",
        "accelerate",
        "pandas",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
