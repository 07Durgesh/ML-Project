# chatbot_core.py
"""
ML-first University Chatbot core (Option B)
- Intent: TF-IDF + SVM
- Entity matching: TF-IDF + cosine similarity (search over DB rows)
- Small rule-based helpers for formatting (list professors/courses, greetings)
- No external generative API dependencies (Gemini removed)
- Accepts explicit deadlines in YYYY-MM-DD format
"""

import sqlite3
import re
from datetime import datetime
import json
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

# Path to your sqlite DB
DB_PATH = "university.db"

# Load training dataset (you already have dataset.py with TRAINING_QUERIES)
from dataset import TRAINING_QUERIES

# ------------------------------------------------------------------
# NLTK bootstrap (downloads if missing)
# ------------------------------------------------------------------
def download_nltk_data():
    required = ["punkt", "wordnet", "stopwords"]
    for pkg in required:
        try:
            if pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


download_nltk_data()

# ------------------------------------------------------------------
# DB helper
# ------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ------------------------------------------------------------------
# NLP Processor
# ------------------------------------------------------------------
class NLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except:
            self.stop_words = set()

    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t.isalnum() and t not in self.stop_words
        ]
        return " ".join(tokens)

# ------------------------------------------------------------------
# Intent Classifier (TF-IDF + SVM)
# ------------------------------------------------------------------
class IntentClassifier:
    def __init__(self, processor: NLPProcessor):
        self.processor = processor
        self.vectorizer = TfidfVectorizer()
        self.model = SVC(kernel="linear", probability=True)
        self._train()

    def _train(self):
        if not TRAINING_QUERIES:
            raise RuntimeError("TRAINING_QUERIES is empty in dataset.py")
        queries, labels = zip(*TRAINING_QUERIES)
        processed = [self.processor.preprocess(q) for q in queries]
        X = self.vectorizer.fit_transform(processed)
        self.model.fit(X, labels)

    def predict(self, text: str):
        p = self.processor.preprocess(text)
        v = self.vectorizer.transform([p])
        label = self.model.predict(v)[0]
        probs = self.model.predict_proba(v)
        confidence = float(np.max(probs))
        return label, confidence

# ------------------------------------------------------------------
# Entity Matcher (search DB rows using TF-IDF + cosine)
# ------------------------------------------------------------------
class EntityMatcher:
    def __init__(self, processor: NLPProcessor):
        self.processor = processor
        self.vectorizer = TfidfVectorizer()

    def find_best_match_sql(self, user_query: str, table: str, columns: list):
        """
        Search the given table over the specified columns.
        Returns (row, score) or (None, 0)
        """
        conn = get_db_connection()
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        conn.close()
        if not rows:
            return None, 0.0

        candidates = []
        for r in rows:
            parts = []
            for c in columns:
                val = r[c] if c in r.keys() and r[c] is not None else ""
                parts.append(str(val))
            candidates.append(" ".join(parts))

        p_query = self.processor.preprocess(user_query)
        p_cands = [self.processor.preprocess(c) for c in candidates]

        # Build TF-IDF matrix (candidates + query)
        try:
            tfidf = self.vectorizer.fit_transform(p_cands + [p_query])
        except ValueError:
            return None, 0.0

        query_vec = tfidf[-1]
        cand_vecs = tfidf[:-1]
        scores = cosine_similarity(query_vec, cand_vecs).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx]) if scores.size > 0 else 0.0

        if best_score < 0.2:
            return None, 0.0
        return rows[best_idx], best_score

    def global_search(self, user_query: str):
        """
        Search across prioritized tables and return first strong match.
        Priority: courses -> professors -> locations
        """
        tables = [
            ("courses", ["name", "code", "dept"]),
            ("professors", ["name", "specialization", "dept", "office"]),
            ("locations", ["name", "building"]),
        ]
        best_row = None
        best_table = None
        best_score = 0.0
        for table, cols in tables:
            row, score = self.find_best_match_sql(user_query, table, cols)
            if row and score > best_score:
                best_row = row
                best_table = table
                best_score = score
        if best_score > 0.2:
            return best_row, best_table
        return None, None

# ------------------------------------------------------------------
# UniversityChatbot (ML-driven + small rules for formatting)
# ------------------------------------------------------------------
class UniversityChatbot:
    def __init__(self):
        self.processor = NLPProcessor()
        self.intent_classifier = IntentClassifier(self.processor)
        self.entity_matcher = EntityMatcher(self.processor)

    # ------------------ DEADLINE HANDLERS ------------------
    def add_deadline_logic(self, message: str):
        """
        Accepts explicit format: "add deadline <title> by YYYY-MM-DD"
        Small rule-only approach for viva reliability.
        """
        m = re.search(r"add deadline\s+(.+?)\s+by\s+(\d{4}-\d{2}-\d{2})", message, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            date_str = m.group(2).strip()
            # Validate date
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return "Please provide date in YYYY-MM-DD format."
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO deadlines (title, due_date, status) VALUES (?, ?, 'pending')",
                (title, date_str),
            )
            conn.commit()
            conn.close()
            return f"✅ Added: {title} (Due: {date_str})"
        return "Please provide deadline like: 'Add deadline AI assignment by 2025-12-10'"

    def mark_deadline_complete(self, message: str):
        m = re.search(r"mark (.+?) as (done|completed)", message, re.IGNORECASE)
        if not m:
            return "Which task do you want to mark as done? Use: Mark <task name> as done"
        q_title = m.group(1).strip()
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM deadlines").fetchall()
        target = None
        for r in rows:
            if q_title.lower() in r["title"].lower():
                target = r
                break
        if not target:
            conn.close()
            return f"Could not find a task matching '{q_title}'."
        conn.execute("UPDATE deadlines SET status='completed' WHERE id=?", (target["id"],))
        conn.commit()
        conn.close()
        return f"🎉 Marked **{target['title']}** as completed!"

    def get_deadline_status(self, filter_type: str):
        conn = get_db_connection()
        today = datetime.now().strftime("%Y-%m-%d")
        if filter_type == "upcoming":
            rows = conn.execute("SELECT * FROM deadlines WHERE due_date >= ? ORDER BY due_date ASC", (today,)).fetchall()
            header = "📅 Upcoming Deadlines:"
        else:
            rows = conn.execute("SELECT * FROM deadlines WHERE due_date < ? ORDER BY due_date DESC", (today,)).fetchall()
            header = "⚠️ Past Deadlines:"
        conn.close()
        if not rows:
            return f"{header}\nNo tasks found."
        out = header + "\n"
        for r in rows:
            mark = "✅" if r["status"] == "completed" else "⏳"
            out += f"{mark} {r['title']} — {r['due_date']}\n"
        return out

    # ------------------ MAIN get_response ------------------
    def get_response(self, message: str):
        if not message or not message.strip():
            return {"response": "Please type a question.", "data": None}

        low = message.lower()

        # quick rule-based greetings (formatting only)
        if any(g in low for g in ["hi", "hello", "hey", "good morning", "good evening"]):
            return {"response": "Hello! I am BU Buddy. I can help with Syllabus, Professors, Locations, and Deadlines.", "data": None}

        # Intent (ML)
        intent, confidence = self.intent_classifier.predict(message)

        # ---------- Direct mapped intents ----------
        if intent == "add_deadline_intent":
            return {"response": self.add_deadline_logic(message), "data": None}
        if intent == "mark_deadline_intent":
            return {"response": self.mark_deadline_complete(message), "data": None}
        if intent == "deadline":
            return {"response": self.get_deadline_status("upcoming"), "data": None}
        if intent == "deadline_history":
            return {"response": self.get_deadline_status("passed"), "data": None}
        if intent == "greeting":
            return {"response": "Hello! I am BU Buddy. How can I assist you?", "data": None}

        # ---------- Professor handling (ML + small rules) ----------
        if intent == "professor":
            # Rule: list all professors if user asks for list
            if "list" in low or "all professors" in low or "list professors" in low:
                conn = get_db_connection()
                rows = conn.execute("SELECT name, dept FROM professors").fetchall()
                conn.close()
                if rows:
                    out = "Here is the list of Professors:\n" + "\n".join([f"- {r['name']} ({r['dept']})" for r in rows])
                    return {"response": out, "data": None}

            # ML entity match: try course -> return assigned prof
            course_row, cscore = self.entity_matcher.find_best_match_sql(message, "courses", ["code", "name"])
            if course_row:
                # find assigned professor row
                prof_name = course_row["assigned_professor"] if "assigned_professor" in course_row.keys() else None
                if prof_name:
                    conn = get_db_connection()
                    prof = conn.execute("SELECT * FROM professors WHERE name = ?", (prof_name,)).fetchone()
                    conn.close()
                    if prof:
                        # Version A formatting: Name + Office + Email
                        return {
                            "response": f"{prof['name']}\nOffice: {prof['office']}\nEmail: {prof['email']}",
                            "data": None,
                        }
                    else:
                        return {"response": f"{course_row['name']} ({course_row['code']}) is assigned to {prof_name} (details not in DB).", "data": None}

            # ML entity match: professor table directly
            prof_row, pscore = self.entity_matcher.find_best_match_sql(message, "professors", ["name", "specialization", "dept"])
            if prof_row:
                return {
                    "response": f"{prof_row['name']}\nOffice: {prof_row['office']}\nEmail: {prof_row['email']}",
                    "data": None,
                }

            # Reverse lookup: if user mentions a prof's exact name, list their courses
            conn = get_db_connection()
            profs = conn.execute("SELECT * FROM professors").fetchall()
            for p in profs:
                if p["name"].lower() in low:
                    courses = conn.execute("SELECT name, code FROM courses WHERE assigned_professor = ?", (p["name"],)).fetchall()
                    conn.close()
                    if courses:
                        clist = ", ".join([f"{c['name']} ({c['code']})" for c in courses])
                        return {"response": f"{p['name']} teaches: {clist}", "data": None}
                    else:
                        return {"response": f"No courses found for {p['name']}.", "data": None}
            conn.close()

            # fallback
            return {"response": "Sorry, I couldn't find that professor. Try asking 'List all professors' or specify the course.", "data": None}

        # ---------- Syllabus ----------
        if intent == "syllabus":
            # list courses rule
            if "list" in low or "all courses" in low:
                conn = get_db_connection()
                rows = conn.execute("SELECT name, code FROM courses").fetchall()
                conn.close()
                if rows:
                    out = "Courses:\n" + "\n".join([f"- {r['name']} ({r['code']})" for r in rows])
                    return {"response": out, "data": None}

            row, score = self.entity_matcher.find_best_match_sql(message, "courses", ["code", "name"])
            if row:
                conn = get_db_connection()
                syl = conn.execute("SELECT * FROM syllabus WHERE course_code = ?", (row["code"],)).fetchone()
                conn.close()
                if syl:
                    return {"response": f"Syllabus for {row['name']}", "data": {"pdf_url": syl["pdf_url"]}}
                return {"response": f"Syllabus entry not found for {row['name']}", "data": None}
            return {"response": "Which course's syllabus would you like? e.g., 'Syllabus for CSET301'", "data": None}

        # ---------- PYQ ----------
        if intent == "pyq":
            row, score = self.entity_matcher.find_best_match_sql(message, "courses", ["code", "name"])
            if row:
                conn = get_db_connection()
                pyq = conn.execute("SELECT * FROM pyqs WHERE course_code = ?", (row["code"],)).fetchone()
                conn.close()
                if pyq:
                    return {"response": f"PYQ for {row['name']}", "data": {"pdf_url": pyq["pdf_url"]}}
                return {"response": f"No PYQ found for {row['name']}", "data": None}
            return {"response": "Which course's PYQ do you want? e.g., 'PYQ for CSET301'", "data": None}

        # ---------- LOCATION ----------
        if intent == "location":
            row, score = self.entity_matcher.find_best_match_sql(message, "locations", ["name", "building"])
            if row:
                return {"response": f"{row['name']} is in {row['building']} (Floor {row['floor']}).", "data": None}
            return {"response": "Which location? e.g., 'Where is the Library?'", "data": None}

        # ---------- Global fallback: try entity search across tables ----------
        row, table_name = self.entity_matcher.global_search(message)
        if row:
            if table_name == "professors":
                return {"response": f"{row['name']}\nOffice: {row['office']}\nEmail: {row['email']}", "data": None}
            if table_name == "courses":
                # attempt to show syllabus/assigned prof info
                conn = get_db_connection()
                prof = conn.execute("SELECT * FROM professors WHERE name = ?", (row["assigned_professor"],)).fetchone()
                syl = conn.execute("SELECT * FROM syllabus WHERE course_code = ?", (row["code"],)).fetchone()
                conn.close()
                out = f"{row['name']} ({row['code']})"
                if prof:
                    out += f"\nTaught by: {prof['name']}"
                if syl:
                    return {"response": out, "data": {"pdf_url": syl["pdf_url"]}}
                return {"response": out, "data": None}
            if table_name == "locations":
                return {"response": f"{row['name']} is in {row['building']} (Floor {row['floor']}).", "data": None}

        # Final fallback
        return {"response": "I couldn't find that information. Please try rephrasing your question.", "data": None}

# If run as script, you can do a quick local test (optional)
if __name__ == "__main__":
    bot = UniversityChatbot()
    tests = [
        "Hi",
        "Who teaches Web Security?",
        "Syllabus for CSET301",
        "Add deadline Final project by 2025-12-01",
        "Show upcoming deadlines",
    ]
    for t in tests:
        print("Q:", t)
        print("A:", bot.get_response(t)["response"])
        print("-----")
