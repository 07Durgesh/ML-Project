# University Information Chatbot
# This final version includes the corrected, robust rule for identifying entities.

import os
from typing import Dict, List, Tuple, Optional
import re
import warnings

# NLP and ML imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# --- IMPORTING all data directly from the dataset file ---
from dataset import (
    TRAINING_QUERIES, COURSES_DATA, PROFESSORS_DATA, SYLLABUS_DATA,
    PYQS_DATA, DEADLINES_DATA, LOCATIONS_DATA
)

# --- NLTK DOWNLOAD SECTION ---
def download_nltk_data():
    """Downloads all necessary NLTK data models if they are not found."""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    for path, model in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK data '{model}' not found. Downloading...")
            nltk.download(model, quiet=True)
            print(f"'{model}' downloaded successfully.")

download_nltk_data()


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {'who', 'what', 'when', 'where', 'which', 'how'}
    
    def preprocess(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]
        return ' '.join(tokens)

class IntentClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=150, ngram_range=(1, 2))
        self.model = SVC(kernel='linear', probability=True, C=1.2)
        self.is_trained = False
    
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        queries, labels = zip(*TRAINING_QUERIES)
        return list(queries), list(labels)

    def train(self):
        X_raw, y = self.prepare_training_data()
        X_processed = [self.preprocessor.preprocess(text) for text in X_raw]
        X_tfidf = self.vectorizer.fit_transform(X_processed)
        self.model.fit(X_tfidf, y)
        self.is_trained = True
        print("Intent classification model trained successfully.")
    
    def predict_intent(self, query: str) -> Tuple[str, float]:
        if not self.is_trained: self.train()
        processed_query = self.preprocessor.preprocess(query)
        X = self.vectorizer.transform([processed_query])
        intent = self.model.predict(X)[0]
        confidence = max(self.model.predict_proba(X)[0])
        return intent, confidence

class UniversityChatbot:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.course_code_to_name = {code: name for code, name, _, _ in COURSES_DATA}
        print("Initializing chatbot...")
        self.classifier.train()
        print("Chatbot ready!")

    # --- IN-MEMORY DATA FETCHING FUNCTIONS ---
    
    def _get_syllabus_from_memory(self, entity: str) -> Optional[Dict]:
        entity_upper = entity.upper()
        # Search by course code
        for s_code, s_content, s_topics, s_pdf in SYLLABUS_DATA:
            if s_code.upper() == entity_upper:
                course_name = self.course_code_to_name.get(s_code, s_code)
                return {'course_name': course_name, 'content': s_content, 'topics': s_topics, 'pdf_url': s_pdf}
        # Search by course name
        for c_code, c_name, _, _ in COURSES_DATA:
            if entity.lower() in c_name.lower():
                for s_code, s_content, s_topics, s_pdf in SYLLABUS_DATA:
                    if s_code == c_code:
                        return {'course_name': c_name, 'content': s_content, 'topics': s_topics, 'pdf_url': s_pdf}
        return None

    def _get_professor_details_from_memory(self, entity: str) -> List[Dict]:
        found_profs = []
        entity_lower = entity.lower()
        for name, email, dept, office, spec in PROFESSORS_DATA:
            if entity_lower in name.lower() or entity_lower in spec.lower():
                found_profs.append({'name': name, 'email': email, 'department': dept, 'office': office, 'specialization': spec})
        return found_profs

    def _get_pyqs_from_memory(self, entity: str) -> List[Dict]:
        found_pyqs = []
        target_code = None
        if entity.upper() in self.course_code_to_name:
            target_code = entity.upper()
        else:
            for code, name in self.course_code_to_name.items():
                if entity.lower() in name.lower():
                    target_code = code
                    break
        
        if target_code:
            course_name = self.course_code_to_name.get(target_code, target_code)
            for p_code, p_year, p_sem, p_qs in PYQS_DATA:
                if p_code == target_code:
                    found_pyqs.append({'course': course_name, 'year': p_year, 'semester': p_sem, 'questions': p_qs})
        return found_pyqs

    def _get_deadlines_from_memory(self) -> List[Dict]:
        deadlines = []
        for title, desc, date, code, type in DEADLINES_DATA:
            course_name = self.course_code_to_name.get(code, "General")
            deadlines.append({'title': title, 'description': desc, 'date': date, 'course': course_name, 'type': type})
        return deadlines
        
    def _get_location_info_from_memory(self, entity: str) -> Optional[Dict]:
        entity_lower = entity.lower()
        for name, building, floor, hours in LOCATIONS_DATA:
            if entity_lower in name.lower():
                return {'name': name, 'building': building, 'floor': floor, 'hours': hours}
        return None

    def _extract_entity(self, query: str, intent: str = '') -> str:
        stop_words = {
            'for', 'of', 'the', 'a', 'an', 'show', 'me', 'what', 'is', 'who', 'get', 
            'find', 'where', 'details', 'contact', 'syllabus', 'professor', 
            'deadlines', 'pyq', 'questions', 'exam', 'paper', 'papers', 'instructor', 'faculty'
        }
        tokens = word_tokenize(query.lower())
        
        course_code_pattern = r'\b[a-zA-Z]{4,8}\d{3}[a-zA-Z]?\b'
        codes = re.findall(course_code_pattern, query.upper())
        if codes: return codes[0]
        
        # Don't return a professor's name here if the rule-based check already did.
        # This function is for finding the subject when the intent is already known.
        filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
        
        if filtered_tokens: return ' '.join(filtered_tokens)
        
        return ''

    def get_response(self, query: str) -> Dict:
        query_lower = query.lower()
        response_data = {'response': "I can help with questions about syllabus, professors, campus locations, and deadlines. How can I assist you?", 'data': None}

        # --- Priority 1: Comprehensive Rule-based Shortcuts ---
        
        # --- CORRECTED RULE FOR PROFESSOR NAMES ---
        for prof_name, _, _, _, _ in PROFESSORS_DATA:
            # Check if the user's query is a substring of the full professor name
            if query_lower in prof_name.lower():
                print(f"Professor name match for '{query_lower}' found in '{prof_name}'. Bypassing ML.")
                return self._handle_professor_query(query_lower, response_data)
        
        for loc_name, _, _, _ in LOCATIONS_DATA:
            if loc_name.lower() in query_lower:
                return self._handle_location_query(loc_name, response_data)

        if 'all professors' in query_lower:
            return self._handle_list_all_professors(response_data)
        if 'deadline' in query_lower or 'due' in query_lower:
            return self._handle_deadline_query(response_data)
        if 'pyq' in query_lower or 'past year' in query_lower or 'exam paper' in query_lower:
            entity = self._extract_entity(query, 'pyq')
            return self._handle_pyq_query(entity, response_data)
        if 'professor' in query_lower or 'faculty' in query_lower or 'instructor' in query_lower:
            entity = self._extract_entity(query, 'professor')
            return self._handle_professor_query(entity, response_data)
        
        for code, name, _, _ in COURSES_DATA:
            if name.lower() in query_lower:
                return self._handle_syllabus_query(name, response_data)

        course_code_pattern = r'\b([a-zA-Z]{4,8}\d{3}[a-zA-Z]?)\b'
        match = re.search(course_code_pattern, query.upper())
        if match:
            entity = match.group(1)
            return self._handle_syllabus_query(entity, response_data)

        # --- Priority 2: ML-based fallback ---
        
        intent, confidence = self.classifier.predict_intent(query)
        if confidence < 0.6: intent = 'general'
            
        entity = self._extract_entity(query, intent)
        
        if not entity and intent in ['syllabus', 'pyq']:
             response_data['response'] = f"It seems you're asking about a {intent}. Could you please specify which course you're interested in?"
             return response_data

        if intent == 'syllabus': return self._handle_syllabus_query(entity, response_data)
        elif intent == 'pyq': return self._handle_pyq_query(entity, response_data)
        else: return response_data

    def _handle_list_all_professors(self, response: Dict) -> Dict:
        prof_names = "\n".join([f"- {name}" for name, _, _, _, _ in PROFESSORS_DATA])
        response['response'] = f"📋 Here is a list of all professors:\n\n{prof_names}\n\nYou can ask for details about any specific professor by name."
        return response

    def _handle_syllabus_query(self, entity: str, response: Dict) -> Dict:
        syllabus = self._get_syllabus_from_memory(entity)
        if syllabus:
            response['data'] = syllabus
            response['response'] = f"📚 Here is the syllabus for **{syllabus['course_name']}**:\n\n**Course Content:**\n{syllabus['content']}\n\n**Topics Covered:**\n{syllabus['topics']}"
        else:
            response['response'] = f"Sorry, I couldn't find the syllabus for '{entity}'."
        return response

    def _handle_professor_query(self, entity: str, response: Dict) -> Dict:
        professors = self._get_professor_details_from_memory(entity)
        if professors:
            response['data'] = professors
            prof_info = "\n\n".join([f"👤 **{p['name']}**\n📧 Email: {p['email']}\n🏢 Office: {p['office']}\n🎓 Specialization: {p['specialization']}" for p in professors])
            response['response'] = f"Here are the details I found:\n\n{prof_info}"
        else:
            response['response'] = f"I couldn't find any professor matching '{entity}'."
        return response

    def _handle_pyq_query(self, entity: str, response: Dict) -> Dict:
        pyqs = self._get_pyqs_from_memory(entity)
        if pyqs:
            response['data'] = pyqs
            course_name_for_title = pyqs[0]['course']
            pyq_info = "\n\n".join([f"📝 **{p['course']} - {p['year']} {p['semester']}**\n{p['questions']}" for p in pyqs])
            response['response'] = f"Here are the past year questions for **{course_name_for_title}**:\n\n{pyq_info}"
        else:
            response['response'] = f"No past year questions found for '{entity}'."
        return response

    def _handle_deadline_query(self, response: Dict) -> Dict:
        deadlines = self._get_deadlines_from_memory()
        if deadlines:
            response['data'] = deadlines
            deadline_info = "\n\n".join([f"📅 **{d['title']}** ({d['course']})\n📝 {d['description']}\n🗓️ Due: {d['date']}" for d in deadlines])
            response['response'] = f"Here are the upcoming deadlines:\n\n{deadline_info}"
        else:
            response['response'] = "No upcoming deadlines found."
        return response

    def _handle_location_query(self, entity: str, response: Dict) -> Dict:
        location = self._get_location_info_from_memory(entity)
        if location:
            response['data'] = location
            response['response'] = (f"📍 Here's the information for the **{location['name']}**:\n\n"
                                  f"**Building:** {location['building']}\n"
                                  f"**Floor:** {location['floor']}\n"
                                  f"**Hours:** {location['hours']}")
        else:
            response['response'] = f"Sorry, I couldn't find a location called '{entity}'."
        return response

