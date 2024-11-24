from setuptools import setup, find_packages

setup(
    name="lstm-time-series-prediction",
    version="0.1.0",
    description="Modelo preditivo de LSTM para previsão de preços de ações.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.115.5",
        "mlflow>=2.17.2",
        "ruff>=0.7.3",
        "scikit-learn>=1.5.2",
        "torch>=2.5.1",
        "uvicorn>=0.32.0",
        "yfinance>=0.2.49",
    ],
)