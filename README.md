# 📄 PDF Q&A with Retrieval Augmented Generation (RAG)

A Streamlit web application that allows users to upload PDF documents and ask questions about their content, leveraging Retrieval Augmented Generation (RAG) to provide accurate, context-aware answers.

## ✨ Features

* **PDF Ingestion:** Upload any PDF document.
* **Intelligent Q&A:** Ask free-form questions and get answers directly from the PDF's content.
* **Source Highlighting (Optional):** See the relevant text snippets from the PDF that informed the answer.
* **User-Friendly UI:** Simple and intuitive web interface built with Streamlit.

## 🚀 Deployed Application

You can access the live version of this application here:
**[Link to your Streamlit Cloud App]((https://pdf-app-rag-app-cncrtsdgyhragn4hcvfuss.streamlit.app/))**


## 🛠️ Technologies Used

* **LangChain:** For building the RAG pipeline (document loading, text splitting, retrieval, QA chain).
* **HuggingFace Transformers:** For the Language Model (Google's Flan-T5-Base) and Tokenizer.
* **Sentence Transformers:** For creating document embeddings (all-MiniLM-L6-v2).
* **PyMuPDFLoader:** For efficient PDF document loading.
* **FAISS:** For fast similarity search and vector storage.
* **Streamlit:** For building the interactive web user interface.

## ⚙️ How to Run Locally

Follow these steps to set up and run the application on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/your-repo-name.git](https://github.com/YOUR_GITHUB_USERNAME/your-repo-name.git)
    cd your-repo-name
    ```
    *(Replace `YOUR_GITHUB_USERNAME` and `your-repo-name` with your actual GitHub username and repository name)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8501`.

    *(Note: The first time you run the app, it will download the embedding and LLM models, which may take a few minutes depending on your internet speed.)*
    
