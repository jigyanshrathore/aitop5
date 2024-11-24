import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from google.cloud import aiplatform

aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

st.cache_data
def load_data():
    questions_df = pd.read_csv("clearfeed_qa_pairs.csv")
    with open("Clearfeed_kb.json", "r") as f:
        doc_data = json.load(f)
    return questions_df, doc_data

def clean_text(input_text):
    input_text = input_text.lower()
    return re.sub(r"[^\w\s]", "", input_text)

def find_top_matches(query, vectorizer, tfidf_matrix, url_list, num_results=5):
    query_vector = vectorizer.transform([clean_text(query)])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-num_results:][::-1]
    return [(url_list[i], similarity_scores[i]) for i in top_indices]

def generate_answer_with_gemini(question, top_results):
    
    context = "\n".join([f"URL: {url}\nContent: {content[:500]}" for url, content in top_results])
    
    prompt = f"""

    Question: {question}

    Context:
    {context}

    """
    
    response = aiplatform.TextGenerationJob.create(
        display_name="gemini-answer-generation",
        prompt=prompt,
        max_output_tokens=200,
        temperature=0.7  
    )
    
    return response.result().text.strip()

def main():
    st.title("ClearFeed Documentation Search with Answer Generation")

    questions_df, doc_data = load_data()

    doc_urls = list(doc_data.keys())
    doc_texts = [clean_text(data['title'] + " " + data['text']) for data in doc_data.values()]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    doc_tfidf_matrix = vectorizer.fit_transform(doc_texts)

    user_query = st.text_input("Enter your question:")
    if user_query:
        top_matches = find_top_matches(user_query, vectorizer, doc_tfidf_matrix, doc_urls)
        top_urls = [url for url, _ in top_matches]
        
        top_contents = [(url, doc_texts[doc_urls.index(url)]) for url in top_urls]

        st.write("Top Matching URLs:")
        for url in top_urls:
            st.write(url)
        
        if st.button("Generate Answer"):
            generated_answer = generate_answer_with_gemini(user_query, top_contents)
            st.write("### Generated Answer:")
            st.write(generated_answer)

if __name__ == "__main__":
    main()
