# Chat-with-PDFs
## Chat PDF

This project allows you to use conversational AI to interact with the content of uploaded PDF files. You can ask questions about the contents of the PDFs, and the model will attempt to answer them based on its understanding of the text.

### Features

* Upload and process multiple PDF files.
* Use a chat interface to ask questions about the content of the PDFs.
* Leverage generative AI for natural language processing and question answering.

### Installation

1. **Prerequisites:** Ensure you have Python 3 and `pip` installed.
2. **Install dependencies:**

   ```bash
   pip install streamlit PyPDF2 langchain faiss transformers
   ```

### Usage

1. **Upload PDFs:**
   - Go to the sidebar menu and select your PDF files using the "Upload your PDF Files and Click on the Submit & Process Button" option.
   - Click the "Submit & Process" button to process the uploaded files.
2. **Ask Questions:**
   - In the chat input box at the bottom of the page, type your question about the processed PDFs.
   - The model will analyze the content and respond to your question.

### How it Works

1. **Processing PDFs:**
   - Uploaded PDFs are first parsed to extract the text content.
   - The extracted text is then split into smaller chunks for efficient processing.
   - A vector store is created using a Generative AI model to encode the text chunks.
2. **Conversational AI:**
   - A pre-trained Chat Generative AI model is used in conjunction with the vector store to answer your questions.
   - The model considers the context of your question and the information obtained from the processed PDFs to provide an answer.

### Contributing

This project is open-source, and contributions are welcome! If you'd like to contribute, please refer to the project's contribution guidelines (if available) or reach out to the project maintainers.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
