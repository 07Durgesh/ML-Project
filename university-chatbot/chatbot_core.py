# chatbot_core.py
"""
University chatbot core (Version A - Full)
 - Intents via SVM + TF-IDF
 - Entity matching via TF-IDF cosine similarity (DB rows)
 - Optional Gemini (google.generativeai) for richer semantic matching / responses
 - Robust deadline handling: regex -> dateparser -> heuristics -> optional AI extraction
 - Professor contact/office minimization: returns only office or email when asked
"""

import sqlite3
import json
import re
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

# NLTK
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

import warnings

# Optional external libs
try:
    import google.generativeai as genai  # optional
except Exception:
    genai = None

try:
    import dateparser
except Exception:
    dateparser = None

# -------------------------------
# CONFIG
# -------------------------------
# Put your API key here if you want Gemini support. If empty or google lib missing, code runs without it.
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"  # replace or leave as-is to operate in fallback-only mode
DB_PATH = 'university.db'

# configure model if available
model = None
if genai and GOOGLE_API_KEY and "YOUR_API_KEY" not in GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        print("Gemini model initialized.")
    except Exception as e:
        print("Gemini initialization failed (will continue without it):", e)
        model = None
else:
    # either genai not installed or key not provided
    model = None

# -------------------------------
# Load training queries
# -------------------------------
try:
    from dataset import TRAINING_QUERIES
except Exception:
    # Minimal fallback if dataset not present (should not happen in your repo)
    TRAINING_QUERIES = [
        ("hi", "greeting"),
        ("who teaches AI", "professor"),
        ("syllabus for CSET301", "syllabus"),
        ("upcoming deadlines", "deadline"),
        ("add deadline", "add_deadline_intent"),
        ("mark task as done", "mark_deadline_intent"),
        ("where is the library", "location"),
        ("previous papers for AI", "pyq"),
    ]

warnings.filterwarnings('ignore')

# -------------------------------
# Ensure NLTK data
# -------------------------------
def ensure_nltk_data():
    resources = [('punkt','tokenizers/punkt'), ('wordnet','corpora/wordnet'), ('stopwords','corpora/stopwords')]
    for res, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass

ensure_nltk_data()

# -------------------------------
# DB Helper
# -------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------------
# NLP Processor
# -------------------------------
class NLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            # fallback tiny stop word set
            self.stop_words = set(['the','is','in','at','a','an','and','or','of','to','for'])

    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        tokens = []
        try:
            raw_tokens = word_tokenize(text)
        except Exception:
            # super-simple fallback
            raw_tokens = re.findall(r"\w+", text)
        for t in raw_tokens:
            if not t.isalnum():
                continue
            if t in self.stop_words:
                continue
            try:
                lem = self.lemmatizer.lemmatize(t)
            except Exception:
                lem = t
            tokens.append(lem)
        return " ".join(tokens)

# -------------------------------
# Intent Classifier (SVM + TF-IDF)
# -------------------------------
class IntentClassifier:
    def __init__(self, processor: NLPProcessor):
        self.processor = processor
        self.vectorizer = TfidfVectorizer()
        self.classifier = SVC(kernel='linear', probability=True)
        self._train()

    def _train(self):
        if not TRAINING_QUERIES:
            return
        queries, labels = zip(*TRAINING_QUERIES)
        processed = [self.processor.preprocess(q) for q in queries]
        X = self.vectorizer.fit_transform(processed)
        self.classifier.fit(X, labels)

    def predict(self, text: str):
        if not text:
            return 'greeting', 0.0
        proc = self.processor.preprocess(text)
        try:
            vec = self.vectorizer.transform([proc])
            pred = self.classifier.predict(vec)[0]
            probs = self.classifier.predict_proba(vec)
            return pred, float(np.max(probs))
        except Exception:
            # fallback heuristic rules
            low = text.lower()
            if any(k in low for k in ['deadline','due','assignment','project','add deadline','remind']):
                return 'add_deadline_intent', 0.6
            if any(k in low for k in ['syllabus','pdf','course','curriculum','syllabus for']):
                return 'syllabus', 0.6
            if any(k in low for k in ['who teaches','professor','teacher','contact','email','office']):
                return 'professor', 0.6
            if any(k in low for k in ['where is','location','floor','building','lab']):
                return 'location', 0.6
            if any(k in low for k in ['pyq','paper','previous','old paper','question paper']):
                return 'pyq', 0.6
            if any(k in low for k in ['hi','hello','hey','help']):
                return 'greeting', 0.9
            return 'unknown', 0.0

# -------------------------------
# Entity Matcher (DB TF-IDF + optional AI fallback)
# -------------------------------
class EntityMatcher:
    def __init__(self, processor: NLPProcessor):
        self.processor = processor
        self.vectorizer = TfidfVectorizer()

    def find_best_match_sql(self, user_query: str, table: str, columns):
        conn = get_db_connection()
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        conn.close()
        if not rows:
            return None, 0.0

        candidates = [" ".join([str(row[col]) for col in columns if row[col] is not None]) for row in rows]
        proc_query = self.processor.preprocess(user_query)
        proc_cands = [self.processor.preprocess(c) for c in candidates]

        try:
            all_text = proc_cands + [proc_query]
            tfidf = self.vectorizer.fit_transform(all_text)
            query_vec = tfidf[-1]
            cand_vecs = tfidf[:-1]
            scores = cosine_similarity(query_vec, cand_vecs).flatten()
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score > 0.2:
                return rows[best_idx], best_score
        except Exception:
            pass

        # fallback to optional semantic AI matcher
        return self.find_semantic_match_ai(user_query, rows, columns)

    def find_semantic_match_ai(self, user_query, rows, columns):
        if not model:
            return None, 0.0
        options = [f"{i}: " + " ".join([str(row[col]) for col in columns if row[col]]) for i, row in enumerate(rows)]
        prompt = (
            f"User query: '{user_query}'\n"
            "Choose the best matching entry and return ONLY the zero-based index. If unsure, return -1.\n\n"
            + "\n".join(options)
        )
        try:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", str(resp)).strip()
            idx = int(re.findall(r"-?\d+", text)[0])
            if 0 <= idx < len(rows):
                return rows[idx], 0.9
        except Exception:
            pass
        return None, 0.0

    def global_search(self, user_query):
        # Prioritize courses -> professors -> locations (helps queries like "who teaches X")
        tables = [
            ('courses', ['name','code','dept']),
            ('professors', ['name','specialization','dept','office']),
            ('locations', ['name','building'])
        ]
        best = None
        best_score = 0.0
        best_table = None
        for table, cols in tables:
            row, score = self.find_best_match_sql(user_query, table, cols)
            if row and score > best_score:
                best_score = score
                best = row
                best_table = table
        if best and best_score > 0.2:
            return best, best_table
        return None, None

# -------------------------------
# Chatbot main logic
# -------------------------------
class UniversityChatbot:
    def __init__(self):
        self.processor = NLPProcessor()
        self.intent_classifier = IntentClassifier(self.processor)
        self.entity_matcher = EntityMatcher(self.processor)

    # ---- AI-based small helpers (optional) ----
    def generate_ai_response(self, user_query, context_data):
        """Use Gemini if available, otherwise simple templated response."""
        if model:
            try:
                prompt = f"You are BU Buddy. Base your concise reply on: {context_data}. User asked: {user_query}"
                resp = model.generate_content(prompt)
                return getattr(resp, "text", str(resp))
            except Exception:
                pass
        # fallback:
        return f"Here is the information I found:\n\n{context_data}"

    def extract_deadline_from_text(self, message):
        """Try multiple strategies to extract {'title','date'} from free text."""
        # 1) explicit regex: "add deadline <title> by YYYY-MM-DD"
        match = re.search(r'add deadline\s+(.+?)\s*by\s*(\d{4}-\d{2}-\d{2})', message, re.IGNORECASE)
        if match:
            return {'title': match.group(1).strip(), 'date': match.group(2)}

        # 2) look for explicit date-like strings with regex and normalize
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', message)
        if date_match:
            title_guess = self._guess_title(message)
            return {'title': title_guess or "Task", 'date': date_match.group(1)}

        # 3) dateparser if available
        if dateparser:
            parsed = dateparser.parse(message, settings={'PREFER_DATES_FROM':'future'})
            if parsed:
                dt = parsed.strftime('%Y-%m-%d')
                title_guess = self._guess_title(message)
                return {'title': title_guess or "Task", 'date': dt}

        # 4) optional AI extraction
        if model:
            try:
                prompt = (
                    f"Today is {datetime.now().strftime('%Y-%m-%d')}. Extract task title and due date from: '{message}'. "
                    "Return ONLY JSON like {'title':'Task Name','date':'YYYY-MM-DD'}. If can't extract date return {}."
                )
                resp = model.generate_content(prompt)
                text = getattr(resp, "text", str(resp))
                cleaned = text.replace('```json','').replace('```','').strip()
                obj = json.loads(cleaned)
                if isinstance(obj, dict) and 'title' in obj and 'date' in obj:
                    return obj
            except Exception:
                pass

        return None

    def _guess_title(self, text):
        # simple heuristic to extract a short meaningful title from a sentence
        t = re.sub(r'(?i)\b(add|deadline|remind|remind me to|i have to|i need to|please|submit|my|on|by|next|this|tomorrow|today|assignment|project|lab|report)\b', ' ', text)
        t = re.sub(r'[^A-Za-z0-9 ]+', ' ', t)
        t = ' '.join(t.split())
        return t.strip().title() if t else None

    # ---- Deadlines handlers ----
    def add_deadline_logic(self, message):
        # Quick check for deadline-related terms
        keywords = ['deadline','due','submit','assignment','project','remind']
        if not any(k in message.lower() for k in keywords):
            return "I couldn't understand the task details. Try: 'Add deadline [Name] by [YYYY-MM-DD]' or mention a date."

        data = self.extract_deadline_from_text(message)
        if data and 'title' in data and 'date' in data:
            # Insert
            conn = get_db_connection()
            conn.execute("INSERT INTO deadlines (title, description, due_date, status) VALUES (?, ?, ?, 'pending')",
                         (data['title'], '', data['date']))
            conn.commit()
            conn.close()
            return f"✅ Added: {data['title']} (Due: {data['date']})"

        # final fallback - ask user to rephrase
        return "I couldn't extract a clear date/title. Please provide 'Add deadline <title> by YYYY-MM-DD' or 'Add deadline <title> by next Friday'."

    def mark_deadline_complete(self, message):
        # Try regex "mark <title> as done"
        match = re.search(r'mark (.+?) as (done|completed)', message, re.IGNORECASE)
        query_title = match.group(1).strip() if match else None

        if not query_title:
            # try AI or heuristics to get title
            # simple fallback: find quoted phrase
            q = re.search(r'["\'](.+?)["\']', message)
            if q:
                query_title = q.group(1).strip()

        if not query_title:
            return "Which task would you like to mark as done? e.g., 'Mark AI Project as completed'"

        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM deadlines").fetchall()
        target = None
        for r in rows:
            if query_title.lower() in r['title'].lower() or r['title'].lower() in query_title.lower():
                target = r
                break
        if target:
            conn.execute("UPDATE deadlines SET status='completed' WHERE id=?", (target['id'],))
            conn.commit()
            conn.close()
            return f"🎉 Marked **{target['title']}** as completed!"
        conn.close()
        return f"Could not find a task matching '{query_title}'. Try a different phrasing."

    def get_deadline_status(self, filter_type='upcoming'):
        conn = get_db_connection()
        today = datetime.now().strftime('%Y-%m-%d')
        if filter_type == 'upcoming':
            rows = conn.execute("SELECT * FROM deadlines WHERE due_date >= ? ORDER BY due_date ASC", (today,)).fetchall()
            header = "📅 Upcoming Deadlines:"
        else:
            rows = conn.execute("SELECT * FROM deadlines WHERE due_date < ? ORDER BY due_date DESC", (today,)).fetchall()
            header = "⚠️ Past/Completed Deadlines:"
        conn.close()
        if not rows:
            return f"{header}\nNo tasks found."
        lines = [header]
        for r in rows:
            mark = "✅" if r['status'] == 'completed' else "⏳"
            lines.append(f"{mark} {r['title']} — {r['due_date']}")
        return "\n".join(lines)

    # ---- Main response pipeline ----
    def get_response(self, message):
        # sanitize
        if not message or not message.strip():
            return {'response': "Please ask something.", 'data': None}

        low = message.lower()

        # quick manual override for show completed/past deadlines
        if any(k in low for k in ['show completed', 'completed work', 'past deadlines', 'show passed', 'passed deadlines', 'task history', 'past']):
            return {'response': self.get_deadline_status('past'), 'data': None}

        intent, conf = self.intent_classifier.predict(message)

        # mapped intents
        if intent == 'add_deadline_intent':
            return {'response': self.add_deadline_logic(message), 'data': None}

        if intent == 'mark_deadline_intent':
            return {'response': self.mark_deadline_complete(message), 'data': None}

        if intent == 'deadline':
            return {'response': self.get_deadline_status('upcoming'), 'data': None}

        if intent == 'deadline_history':
            return {'response': self.get_deadline_status('past'), 'data': None}

        if intent == 'greeting':
            return {'response': "Hello! I am BU Buddy. I can help with Syllabus, Professors, Locations, PYQs and Deadlines.", 'data': None}

        # PROFESSOR intent handling (prioritize course -> professor)
        context_info = None
        if intent == 'professor':
            # list professors
            if any(x in low for x in ['list', 'all', 'everyone', 'show']) and ('professor' in low or 'professors' in low):
                conn = get_db_connection()
                rows = conn.execute("SELECT name, dept FROM professors").fetchall()
                conn.close()
                if rows:
                    out = "Here is the list of Professors:\n" + "\n".join([f"- {r['name']} ({r['dept']})" for r in rows])
                    return {'response': out, 'data': None}

            # try course match first (e.g., "who teaches Human Computer Interaction")
            course_row, score = self.entity_matcher.find_best_match_sql(message, 'courses', ['code','name'])
            if course_row:
                conn = get_db_connection()
                prof = conn.execute("SELECT * FROM professors WHERE name = ?", (course_row['assigned_professor'],)).fetchone()
                conn.close()
                if prof:
                    # if user asked about office/email specifically, return that only
                    if any(w in low for w in ['office', 'where does', 'cabin', 'room']):
                        return {'response': f"{prof['name']} - Office: {prof['office']}", 'data': None}
                    if any(w in low for w in ['email', 'contact', 'how to contact', 'phone', 'contact details']):
                        return {'response': f"{prof['name']} - Email: {prof['email']}", 'data': None}
                    # otherwise return short contact info
                    return {'response': f"Name: {prof['name']}, Office: {prof['office']}, Email: {prof['email']}", 'data': None}
                else:
                    return {'response': f"Course: {course_row['name']} ({course_row['code']}).", 'data': None}

            # match on professor table
            prof_row, pscore = self.entity_matcher.find_best_match_sql(message, 'professors', ['name','specialization','dept'])
            if prof_row:
                # if user only asked for office or contact return only that
                if any(w in low for w in ['office', 'where does', 'cabin', 'room']):
                    return {'response': f"{prof_row['name']} - Office: {prof_row['office']}", 'data': None}
                if any(w in low for w in ['email', 'contact', 'how to contact', 'phone', 'contact details']):
                    return {'response': f"{prof_row['name']} - Email: {prof_row['email']}", 'data': None}
                return {'response': f"Name: {prof_row['name']}, Office: {prof_row['office']}, Email: {prof_row['email']}, Spec: {prof_row['specialization']}", 'data': None}

            # reverse lookup by explicit professor name: return their courses
            conn = get_db_connection()
            profs = conn.execute("SELECT * FROM professors").fetchall()
            for p in profs:
                if p['name'].lower() in low:
                    courses = conn.execute("SELECT * FROM courses WHERE assigned_professor = ?", (p['name'],)).fetchall()
                    conn.close()
                    if courses:
                        course_list = ", ".join([f"{c['name']} ({c['code']})" for c in courses])
                        return {'response': f"{p['name']} teaches: {course_list}", 'data': None}
                    else:
                        return {'response': f"No courses found for {p['name']}.", 'data': None}
            conn.close()

        # SYLLABUS intent
        if intent == 'syllabus':
            if any(x in low for x in ['list', 'all', 'courses']):
                conn = get_db_connection()
                rows = conn.execute("SELECT name, code FROM courses").fetchall()
                conn.close()
                if rows:
                    text = "\n".join([f"- {r['name']} ({r['code']})" for r in rows])
                    return {'response': "Here are the available courses:\n" + text, 'data': None}
            # specific course
            row, score = self.entity_matcher.find_best_match_sql(message, 'courses', ['code','name'])
            if row:
                conn = get_db_connection()
                syl = conn.execute("SELECT * FROM syllabus WHERE course_code = ?", (row['code'],)).fetchone()
                conn.close()
                if syl:
                    return {'response': f"Syllabus for {row['name']}.", 'data': {'pdf_url': syl['pdf_url']}}
                return {'response': f"Course {row['name']} ({row['code']}) found, but no syllabus PDF available.", 'data': None}
            return {'response': "Which course's syllabus would you like? e.g., 'Syllabus for CSET301'", 'data': None}

        # PYQ intent
        if intent == 'pyq':
            row, score = self.entity_matcher.find_best_match_sql(message, 'courses', ['code','name'])
            if row:
                conn = get_db_connection()
                pyq = conn.execute("SELECT * FROM pyqs WHERE course_code = ?", (row['code'],)).fetchone()
                conn.close()
                if pyq:
                    return {'response': f"Here is the PYQ for {row['name']}.", 'data': {'pdf_url': pyq['pdf_url']}}
                return {'response': f"No PYQ found for {row['name']}.", 'data': None}
            return {'response': "Which course PYQ do you want? e.g., 'PYQ for CSET301'", 'data': None}

        # LOCATION intent
        if intent == 'location':
            row, score = self.entity_matcher.find_best_match_sql(message, 'locations', ['name','building'])
            if row:
                return {'response': f"{row['name']} is in {row['building']} (Floor {row['floor']}).", 'data': None}
            return {'response': "I couldn't find that location. Try full name like 'Computer Lab 1' or 'Library'.", 'data': None}

        # GLOBAL fallback using entity matcher
        row, table_name = self.entity_matcher.global_search(message)
        if row:
            if table_name == 'professors':
                # If user asked for only office/email, check for those words
                if any(w in low for w in ['office','cabin','room','where does']):
                    return {'response': f"{row['name']} - Office: {row['office']}", 'data': None}
                if any(w in low for w in ['email','contact','how to contact']):
                    return {'response': f"{row['name']} - Email: {row['email']}", 'data': None}
                return {'response': self.generate_ai_response(message, f"Professor: {row['name']}, Office: {row['office']}, Email: {row['email']}"), 'data': None}
            elif table_name == 'courses':
                conn = get_db_connection()
                prof_row = conn.execute("SELECT * FROM professors WHERE name = ?", (row['assigned_professor'],)).fetchone()
                conn.close()
                if prof_row:
                    return {'response': self.generate_ai_response(message, f"Course: {row['name']} ({row['code']}). Taught by {prof_row['name']}."), 'data': None}
                return {'response': self.generate_ai_response(message, f"Course: {row['name']} ({row['code']})."), 'data': None}
            elif table_name == 'locations':
                return {'response': self.generate_ai_response(message, f"Location: {row['name']} in {row['building']}"), 'data': None}

        # final fallback
        return {'response': "I couldn't find that information. Please try rephrasing or ask about courses, professors, locations or deadlines.", 'data': None}

# End of chatbot_core.py
