import sqlite3
from datetime import datetime
import os

DB_FILE = "university.db"

# ===============================
# 1. CREATE TABLES
# ===============================
def create_tables(conn):
    cur = conn.cursor()

    # Enable foreign keys
    cur.execute("PRAGMA foreign_keys = ON;")

    # courses table now includes an assigned_professor column (TEXT)
    cur.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        code TEXT PRIMARY KEY,
        name TEXT,
        credits INTEGER,
        dept TEXT,
        assigned_professor TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS professors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        email TEXT,
        dept TEXT,
        office TEXT,
        specialization TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS syllabus (
        course_code TEXT PRIMARY KEY,
        content TEXT,
        topics TEXT,
        pdf_url TEXT,
        FOREIGN KEY(course_code) REFERENCES courses(code) ON DELETE CASCADE
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS pyqs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_code TEXT,
        year TEXT,
        semester TEXT,
        pdf_url TEXT,
        FOREIGN KEY(course_code) REFERENCES courses(code) ON DELETE CASCADE
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        building TEXT,
        floor INTEGER,
        hours TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS deadlines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        due_date DATE,
        status TEXT DEFAULT 'pending'
    )
    ''')

    conn.commit()


# ===============================
# 2. SEED DATA (Professors, Courses, Syllabus, PYQs, etc.)
# ===============================
def seed_data(conn):
    cur = conn.cursor()

    # Clear existing rows (fresh seed)
    cur.execute("DELETE FROM pyqs")
    cur.execute("DELETE FROM syllabus")
    cur.execute("DELETE FROM courses")
    cur.execute("DELETE FROM professors")
    cur.execute("DELETE FROM locations")
    cur.execute("DELETE FROM deadlines")
    conn.commit()

    # -------------------------------
    # 12 Professors
    # -------------------------------
    professors = [
        ("Dr. Vikram Singh",     "vikram.singh@university.edu", "Computer Science", "D-401", "Artificial Intelligence"),
        ("Prof. Priya Sharma",   "priya.sharma@university.edu", "Computer Science", "A-101", "Data Structures"),
        ("Dr. Rajesh Kumar",     "rajesh.kumar@university.edu", "Computer Science", "B-205", "Algorithms"),
        ("Prof. Neha Verma",     "neha.verma@university.edu", "Mathematics",       "C-110", "Calculus"),
        ("Dr. Sandeep Reddy",    "sandeep.reddy@university.edu","Electrical Eng",   "E-302", "Signals"),
        ("Prof. Anjali Mehta",   "anjali.mehta@university.edu", "Mechanical Eng",    "M-210", "Thermodynamics"),
        ("Dr. Arjun Rao",        "arjun.rao@university.edu",     "Physics",           "P-102", "Quantum Mechanics"),
        ("Prof. Kavita Gupta",   "kavita.gupta@university.edu",  "Chemistry",         "CH-14", "Organic Chemistry"),
        ("Dr. Amit Patel",       "amit.patel@university.edu",    "Business",          "BIZ-11","Marketing"),
        ("Prof. Rahul Joshi",    "rahul.joshi@university.edu",   "Psychology",        "PSY-7", "Cognitive Science"),
        ("Dr. Emily Johnson",    "emily.johnson@university.edu", "Computer Science",  "CS-220","Web Security"),
        ("Prof. Robert Brown",   "robert.brown@university.edu",  "Computer Science",  "CS-330","Database Systems"),
    ]

    cur.executemany(
        'INSERT INTO professors (name, email, dept, office, specialization) VALUES (?,?,?,?,?)',
        professors
    )
    conn.commit()

    # Fetch list of professors for assignment
    cur.execute("SELECT name FROM professors ORDER BY id")
    prof_rows = [r[0] for r in cur.fetchall()]

    # -------------------------------
    # 25 Courses (with assigned professors)
    # -------------------------------
    course_list = [
        ("CSET301", "AI & Machine Learning", 4, "Computer Science"),
        ("CSET302", "Data Structures and Algorithms", 4, "Computer Science"),
        ("CSET303", "Operating Systems", 3, "Computer Science"),
        ("CSET304", "Database Systems", 3, "Computer Science"),
        ("CSET305", "Web Security", 3, "Computer Science"),
        ("MATH201", "Calculus II", 3, "Mathematics"),
        ("MATH202", "Linear Algebra", 3, "Mathematics"),
        ("PHYS101", "Physics I", 3, "Physics"),
        ("PHYS102", "Quantum Mechanics", 3, "Physics"),
        ("MECH201", "Thermodynamics", 3, "Mechanical Eng"),
        ("EE201", "Signals and Systems", 3, "Electrical Eng"),
        ("CHEM101", "Organic Chemistry I", 3, "Chemistry"),
        ("BUS101", "Principles of Marketing", 3, "Business"),
        ("PSY101", "Introduction to Cognitive Science", 3, "Psychology"),
        ("CS350", "Machine Learning Lab", 2, "Computer Science"),
        ("CS360", "Natural Language Processing", 3, "Computer Science"),
        ("CS370", "Computer Networks", 3, "Computer Science"),
        ("CS380", "Software Engineering", 3, "Computer Science"),
        ("CSET401", "Advanced Algorithms", 3, "Computer Science"),
        ("CSET402", "Distributed Systems", 3, "Computer Science"),
        ("CSET403", "Information Retrieval", 3, "Computer Science"),
        ("EE301", "Digital Electronics", 3, "Electrical Eng"),
        ("MATH301", "Probability & Statistics", 3, "Mathematics"),
        ("CS390", "Human-Computer Interaction", 3, "Computer Science"),
        ("CS395", "Capstone Project", 4, "Computer Science"),
    ]

    assigned_courses = []
    for i, c in enumerate(course_list):
        prof = prof_rows[i % len(prof_rows)]
        code, name, credits, dept = c
        assigned_courses.append((code, name, credits, dept, prof))

    cur.executemany(
        'INSERT INTO courses (code, name, credits, dept, assigned_professor) VALUES (?,?,?,?,?)',
        assigned_courses
    )
    conn.commit()

    # -------------------------------
    # Syllabus & PYQs
    # -------------------------------
    syllabus_rows = []
    pyq_rows = []

    year_choices = ["2019", "2020", "2021", "2022", "2023", "2024"]
    sem_choices = ["Fall", "Spring"]

    for code, name, credits, dept, prof in assigned_courses:
        syl_pdf = f"pdfs/{code}_Syllabus.pdf"
        pyq_pdf = f"pdfs/{code}_PYQ.pdf"

        syllabus_rows.append(
            (code, f"Syllabus content for {name}", f"Core topics of {name}", syl_pdf)
        )

        pyq_rows.append(
            (code, year_choices[hash(code) % len(year_choices)],
             sem_choices[hash(code + 'x') % len(sem_choices)],
             pyq_pdf)
        )

    cur.executemany('INSERT INTO syllabus (course_code, content, topics, pdf_url) VALUES (?,?,?,?)', syllabus_rows)
    cur.executemany('INSERT INTO pyqs (course_code, year, semester, pdf_url) VALUES (?,?,?,?)', pyq_rows)
    conn.commit()

    # -------------------------------
    # Locations
    # -------------------------------
    locations = [
        ("Library", "Main Block", 2, "8 AM - 10 PM"),
        ("Cafeteria", "Student Center", 0, "8 AM - 8 PM"),
        ("Computer Lab 1", "Tech Park", 1, "9 AM - 6 PM"),
        ("Auditorium", "Main Block", 1, "9 AM - 9 PM"),
    ]

    cur.executemany('INSERT INTO locations (name, building, floor, hours) VALUES (?,?,?,?)', locations)
    conn.commit()

    # -------------------------------
    # Sample Deadlines
    # -------------------------------
    deadlines = [
        ("Submit AI Project", "Submit via LMS", "2025-12-01", "pending"),
        ("Capstone Proposal", "Upload proposal", "2025-11-25", "pending"),
    ]

    cur.executemany('INSERT INTO deadlines (title, description, due_date, status) VALUES (?,?,?,?)', deadlines)
    conn.commit()

    # Verification Print
    for t in ["courses", "professors", "syllabus", "pyqs", "locations", "deadlines"]:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"{t}: {cur.fetchone()[0]} entries")


# ===============================
# MAIN ENTRY POINT
# ===============================
def main():
    conn = sqlite3.connect(DB_FILE)
    create_tables(conn)
    seed_data(conn)
    conn.close()
    print("\n✅ Database setup complete. Place your PDFs inside ./static/pdfs/")


if __name__ == "__main__":
    main()
