# Muhammad Owais Raza Qadri
# Create a Chatbot for FAQS.
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download

def download_nltk_resources():
    """Ensure all required NLTK resources are available"""
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk_download(resource)

class FAQChatbot:
    def __init__(self, faqs):
        """Initialize the chatbot with FAQs"""
        
        download_nltk_resources()
        
     
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.faqs = faqs
        self.questions = [faq['question'] for faq in faqs]
        
       
        def tokenizer_func(text):
            return self.preprocess_text(text)
            
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer_func)
        self.vectorizer.fit_transform(self.questions)

    def preprocess_text(self, text):
        """Clean and preprocess text for vectorization"""
        
        text = text.lower()
        
      
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        
        
        tokens = nltk.word_tokenize(text)
        
       
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return processed_tokens

    def get_response(self, query):
        """Get best matching answer for user query"""
        try:
        
            query_vec = self.vectorizer.transform([query])
            
            
            question_vecs = self.vectorizer.transform(self.questions)
            
            
            similarities = cosine_similarity(query_vec, question_vecs)
            
         
            best_match_idx = np.argmax(similarities)
            
            
            if similarities[0, best_match_idx] > 0.2:
                return self.faqs[best_match_idx]['answer']
            return "I'm not sure I understand. Could you rephrase your question?"
        except Exception as e:
            print(f"Error in response generation: {e}")
            return "Sorry, I encountered an error processing your request."

# Sample FAQs
sample_faqs = [
    {
        'question': "What is my Name",
        'answer': "Your Name is Muhammad Owais Raza Qadri."
    },
    {
        'question': "What is my age and Gender",
        'answer': "Your age is 22 Years and You are Male."
    }
]

def main():
    print("Initializing FAQ Chatbot...")
    chatbot = FAQChatbot(sample_faqs)
    
    print("\nFAQ Chatbot: Hello! Ask me anything about our services.")
    print("Type 'exit' to end the conversation.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("FAQ Chatbot: Goodbye!")
                break
                
            if not user_input:
                print("FAQ Chatbot: Please enter a question.")
                continue
                
            response = chatbot.get_response(user_input)
            print(f"FAQ Chatbot: {response}\n")
            
        except KeyboardInterrupt:
            print("\nFAQ Chatbot: Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()