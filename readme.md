# AI Underwriter Pro 🛡️

AI Underwriter Pro is a lightweight, serverless-friendly web application designed to automate and streamline the insurance risk analysis process. By leveraging Large Language Models (LLMs), the application ingests standard Underwriting Guidelines and applicant Data Forms, cross-references the rule sets, and instantly extracts critical risk factors.

## 🚀 Features
* **Automated Document Parsing:** Securely processes and extracts text from uploaded PDF guidelines and application forms using LangChain.
* **Intelligent Chunking:** Utilizes advanced text-splitting algorithms to handle large enterprise documents without losing contextual meaning.
* **AI-Powered Risk Extraction:** Employs OpenAI's `gpt-3.5-turbo` to perform deterministic, highly factual risk assessments based *strictly* on the provided guidelines.
* **Enterprise-Ready UI:** A clean, responsive, wide-layout interface built with Streamlit, featuring expandable risk reports and dynamic status indicators.
* **Cloud-Optimized Architecture:** Refactored to offload heavy ML processing to APIs, keeping the container footprint well under 1GB for seamless deployment on free-tier or standard cloud hosting environments.

## 🛠️ Technology Stack
* **Frontend/Framework:** [Streamlit](https://streamlit.io/)
* **LLM Engine:** [OpenAI API](https://platform.openai.com/)
* **Orchestration & Document Loaders:** [LangChain](https://python.langchain.com/)
* **PDF Extraction:** `pypdf`
* **Language:** Python 3.9+

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ai-underwriter-pro.git](https://github.com/yourusername/ai-underwriter-pro.git)
   cd ai-underwriter-pro
