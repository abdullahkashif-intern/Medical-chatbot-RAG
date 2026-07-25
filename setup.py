from setuptools import setup, find_packages

setup(
    name='Medical-chatbot-RAG',
    version='0.0.1',
    author='Abdullah Kashif',
    author_email='[EMAIL_ADDRESS]',
    packages=find_packages(),
    install_requires=[
        'langchain',
        'langchain-community',
        'langchain-core',
        'langchain-openai',
        'sentence-transformers',
        'pypdf',
        'python-dotenv',
        'langchain-pinecone',
        '-e .',
    ],
)