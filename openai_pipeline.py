# openai_pipeline.py

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()  
    return pages  # Return list of pages directly


def openai_pipeline(file_path, api_key, max_tokens=300):
    """
    Summarize a PDF file page by page using OpenAI API.
    Returns two lists: summaries and titles, one per page.
    """
    openai.api_key = api_key
    pages = file_preprocessing(file_path)
    summaries = []
    titles = []

    for i, page in enumerate(pages):
        # Split page into chunks to avoid exceeding token limit
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
        chunks = text_splitter.split_documents([page])
        page_text = "".join([chunk.page_content for chunk in chunks])

        #Generate summary
        summary_prompt = f"""
        You are a helpful teaching assistant. Your task is to explain and summarize the following text in a way that a student can easily understand. Make sure to:

        1. Include all important concepts and key points.
        2. Explain technical terms in simple language.
        3. Provide short examples or analogies if needed.
        4. Keep the explanation concise but comprehensive.

        Text:
        {page_text}

        Please return the summary as clear paragraph suitable for studying, not as bullet points.
        """
        summary_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )

        summary_text = summary_response.choices[0].message.content.strip()
        summaries.append(summary_text)

        #Generate title
        title_prompt = f"""
        Create a concise, descriptive title for the following content suitable for a presentation slide:

        {page_text}
        """
        title_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": title_prompt}],
            temperature=0.5,
            max_tokens=20
        )
        title_text = title_response.choices[0].message.content.strip()
        titles.append(title_text)

    return summaries, titles
