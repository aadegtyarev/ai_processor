
from setuptools import setup, find_packages

setup(
    name="ai_processor",
    version="1.0.1",
    description="A library for processing text with neural networks using chunked prompts.",
    author="Alexander Degtyarev",
    author_email="adegtyarev.ap@gmail.com",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.9',
)
