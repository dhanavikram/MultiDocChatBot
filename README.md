# Chat with Multiple Documents

A Mistral 7B based RAG chatbot to chat with your documents. Accepts PDFs, Word Documents and .txt files.

The model is not uploaded to save space. To run,

1. Git clone this repo
2. Create a models folder.
3. Download the model from [this]("https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main) link.
4. *Optional*: Download the model of your choice and rename it accordingly in line 64 of `app.py`
5. Run `streamlit run app.py` in the command line.
6. Upload your docs and get answers for your questions.