# PCL-Chat bot

This is a repository for our summer school project 2024 at university of applied sciences Kalrsurhe. The goal of this project is to create a retrieval-augmented generation (RAG) chatbot that can answer questions about the Point Cloud Library (PCL) documentation. The chatbot will be able to answer questions about the [Point Cloud Library](https://pointclouds.org/documentation/) by retrieving relevant information from the documentation and generating an answer based on the retrieved information.

## Features

- **Web Scraping**: Utilizes BeautifulSoup to parse HTML content of the documentation and extract relevant information.
- **Data Processing**: Employs pandas for data manipulation and storage.
- **Document Analysis**: Analyzes different types of documentation elements such as classes, functions, and descriptions.
- **CSV Export**: Outputs the processed data into a CSV file for easy access and further analysis.
- **Streamlit Integration**: Provides a user-friendly interface to interact with the processed data.
- **Retrieval-Augmented Generation (RAG)**: Implements a RAG pipeline using Haystack with HyDE (with an option to alternatively use HyQE) to generate answers to user questions based on the processed data.


## Getting Started

### Dependencies

* The project requires Python 3.10 or later and depends on
	- `beautifulsoup4`
	- `pandas`
	- `streamlit`
	- `haystack-ai`
	- `qdrant-haystack`
	- `pypdf`
	- `markdown-it-py`
	- `sentence-transformers`
	- `cryptography`
	- `langfuse-haystack`
	- `langdetect`

* The project also requires the following tools:
	- `ollama`
	- `docker`

* While this app can be run on any operating system supporting the above dependencies and tools, it has been tested and instructions have been provided for Ubuntu 22.04.

### Installation

1. **install poetry via pipx**:
	```bash
	pip3 install pipx
	pipx install poetry
	```
2. **Install ollama**:
	```bash
	cd ~
	curl -fsSL https://ollama.com/install.sh | sh
	```
3. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/rag-project.git
    cd rag-project
    ```
4. **Setup your virtual environment**:
	```bash
	python3 -m venv .venv
	```

5. **install dependencies from repository root**:
	```bash
	source .venv/bin/activate
	poetry install
	```

6. **Setup environment variables for tracing via Langfuse**:
	```bash
	echo "export LANGFUSE_SECRET_KEY=<your-secret-key> >> ~/.bashrc"
	echo "export LANGFUSE_PUBLIC_KEY=<your-public-key> >> ~/.bashrc"
	```

### Running the RAG app
1. **Pull the latest version of llama3.1**:
	```bash
	ollama pull llama3.1
	```

2. **Start your local qdrant intance**:
	```bash
	docker run -p 6333:6333 -p 6334:6334 \
		-v ~/qdrant_storage:/qdrant/storage:z \
		qdrant/qdrant
	```
3. **Activate the virtual environment**:
	```bash
	source .venv/bin/activate
	```

4. **From the `src` folder of the repository, run the app**:
	```bash
	cd src
	streamlit run main.py
	```

5. **An instance of your browser should open, to look something like this**:
![RAG App Screenshot](docs/Chatbot-main_page.png)