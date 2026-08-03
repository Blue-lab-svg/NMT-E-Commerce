from setuptools import setup, find_packages  # type: ignore

setup(
    name="lemt",
    version="0.1.0",
    description="LEMT - Language E-commerce Machine Translation pipeline",
    author="Blue Lab",
    author_email="info@bluelab.com",
    url="https://github.com/Blue-lab-svg/LEMT",
    package_dir={"": "src"},              
    packages=find_packages(where="src"), 
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "sacrebleu",
        "fastapi",
        "uvicorn",
        "pytest",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
