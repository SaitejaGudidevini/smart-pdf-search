from setuptools import setup, find_packages

setup(
    name='mayan-rag-chat',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fastembed>=0.7.4',
        'numpy',
        'qdrant-client>=1.13.0',
        'rank-bm25>=0.2.2',
        'langchain-text-splitters>=0.3.8',
        'httpx>=0.28.0',
    ],
)
