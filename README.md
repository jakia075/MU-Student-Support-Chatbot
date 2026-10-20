# MU Student Support Chatbot

Trilingual (English, Bangla, Banglish) retrieval-augmented chatbot for Metropolitan University, Sylhet.
BSc thesis by **Jakia Habiba** and **Mst. Samia Begum**, supervised by **Samia Rahman Rima**,
Department of CSE, Metropolitan University, Sylhet.

## Files
- `MUBot_v5_RAG_vs_LLM_colab.ipynb` – full evaluation (LLM-only vs RAG) and the web app; open in Google Colab with a T4 GPU and run all cells.
- `mu_rag.py` – retrieval, answer-or-refuse decision and guarded answering (used by the notebook and the app).
- `app.py` – Streamlit web application.
- `create_mu_survey_form.gs` – Google Apps Script that creates the user-study questionnaire.

The dataset files (`MUBot_v5_*.csv`) are not public because they contain contact details; they are available from the authors on request.
