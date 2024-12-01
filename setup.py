from setuptools import setup, find_packages

setup(
    name="api",
    version="0.1.0",
    description="Modelo preditivo de LSTM para previsão de preços de ações.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
