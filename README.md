ClearFeed Documentation Search with AI Answer Generation
This project is a documentation search system enhanced with AI-powered answer generation using Google’s Gemini API. It allows users to input questions related to ClearFeed documentation, retrieves the most relevant URLs, and generates a concise answer using advanced AI techniques.

Features
Search System:

Uses TF-IDF vectorization to retrieve the top 5 relevant documentation URLs for a user query.
Ensures efficient and accurate search functionality.
AI Answer Generation:

Leverages Google Gemini API to generate concise answers from the top retrieved documentation URLs.
Interactive Web Interface:

Built with Streamlit for easy interaction.
Users can enter questions, view the top matching URLs, and generate answers with a single click.
Project Workflow
Data Loading:

Load the ClearFeed question-answer dataset (clearfeed_qa_pairs.csv) and the ClearFeed documentation knowledge base (Clearfeed_kb.json).
Data Preprocessing:

Clean and preprocess text by removing punctuation and converting it to lowercase.
Search System:

Convert documentation content into TF-IDF vectors.
Retrieve the top 5 most relevant URLs using cosine similarity.
Answer Generation:

Use the Google Gemini API to generate answers based on the top 5 matching URLs and their content.
Streamlit Interface:

Interactive app where users can:
Enter a question.
View the top matching URLs.
Generate an AI-powered answer.
How to Use
1. Prerequisites
Python 3.7 or higher installed on your system.
Google Cloud SDK and a valid Google Cloud Project with Generative AI Support (Gemini) API enabled.
