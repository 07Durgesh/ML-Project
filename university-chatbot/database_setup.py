import sqlite3
import random
from datetime import datetime, timedelta

def create_database():
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()

    # --- 1. CREATE TABLES ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        code TEXT PRIMARY KEY,
        name TEXT,
        credits INTEGER,
        dept TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS professors (
        name TEXT,
        email TEXT,
        dept TEXT,
        office TEXT,
        specialization TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS syllabus (
        course_code TEXT,
        content TEXT,
        topics TEXT,
        pdf_url TEXT,
        FOREIGN KEY(course_code) REFERENCES courses(code)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pyqs (
        course_code TEXT,
        year TEXT,
        semester TEXT,
        pdf_url TEXT,
        FOREIGN KEY(course_code) REFERENCES courses(code)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS locations (
        name TEXT,
        building TEXT,
        floor INTEGER,
        hours TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS deadlines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        due_date DATE,
        status TEXT DEFAULT 'pending'
    )
    ''')

    # --- 2. GENERATE VAST DATA (500+ Entries) ---
    
    print("Generating vast dataset...")

    # -- Data Pools --
    departments = ['Computer Science', 'Mathematics', 'Physics', 'Mechanical Eng', 'Electrical Eng', 'Business', 'Psychology', 'Chemistry']
    first_names = ['Vikram', 'Priya', 'Rajesh', 'Amit', 'Sneha', 'Anjali', 'Rahul', 'David', 'Sarah', 'John', 'Emily', 'Michael', 'Robert', 'Linda', 'William', 'Elizabeth', 'Arjun', 'Aditi', 'Kavita', 'Sanjay']
    last_names = ['Singh', 'Sharma', 'Kumar', 'Patel', 'Gupta', 'Verma', 'Mishra', 'Reddy', 'Nair', 'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez']
    course_types = ['Introduction to', 'Advanced', 'Principles of', 'Fundamentals of', 'Applied', 'Theoretical']
    topics_pool = ['AI', 'Machine Learning', 'Thermodynamics', 'Quantum Mechanics', 'Data Structures', 'Algorithms', 'Calculus', 'Linear Algebra', 'Organic Chemistry', 'Marketing', 'Finance', 'Cognitive Science']
    buildings = ['Main Block', 'Science Hub', 'Tech Park', 'Library Building', 'Student Center', 'Innovation Lab', 'Admin Block']
    
    courses = []
    professors = []
    syllabi = []
    pyqs = []
    deadlines = []
    locations = []

    # A. Generate 150 Courses
    for i in range(150):
        dept = random.choice(departments)
        code_prefix = dept[:3].upper()
        code = f"{code_prefix}{100 + i}"
        topic = random.choice(topics_pool)
        prefix = random.choice(course_types)
        name = f"{prefix} {topic} {i+1}"
        credits = random.choice([2, 3, 4])
        courses.append((code, name, credits, dept))

        # B. Generate Syllabus & PYQ for each course (Total 300 entries here)
        syllabi.append((code, f"Full syllabus for {name}", f"Unit 1-5: {topic} concepts", f"pdfs/{code}_Syllabus.pdf"))
        pyqs.append((code, str(random.randint(2018, 2024)), random.choice(['Fall', 'Spring']), f"pdfs/{code}_PYQ.pdf"))

    # C. Generate 100 Professors
    for i in range(100):
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        name = f"Dr. {fname} {lname}"
        dept = random.choice(departments)
        email = f"{fname.lower()}.{lname.lower()}@university.edu"
        office = f"{random.choice(['A', 'B', 'C', 'D'])}-{random.randint(100, 500)}"
        spec = random.choice(topics_pool)
        professors.append((name, email, dept, office, spec))

    # D. Generate 100 Deadlines (Past & Future)
    for i in range(100):
        topic = random.choice(topics_pool)
        task_types = ['Assignment', 'Project', 'Lab Report', 'Mid-Term Exam', 'Final Presentation']
        title = f"{topic} {random.choice(task_types)} {i+1}"
        
        # Random date within +/- 60 days
        days_offset = random.randint(-60, 60)
        date = (datetime.now() + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        status = 'completed' if days_offset < 0 else 'pending'
        
        deadlines.append((title, f"Submit via portal", date, status))

    # E. Generate 60 Locations
    location_types = ['Lab', 'Classroom', 'Seminar Hall', 'Cafeteria', 'Office', 'Gym', 'Store']
    for i in range(60):
        l_type = random.choice(location_types)
        name = f"{l_type} {i+101}"
        building = random.choice(buildings)
        floor = random.randint(0, 5)
        hours = "9 AM - 5 PM" if l_type == 'Office' else "8 AM - 10 PM"
        locations.append((name, building, floor, hours))

    # --- 3. INSERT DATA ---
    
    # Clear old data to avoid duplicates on re-run
    cursor.execute("DELETE FROM courses")
    cursor.execute("DELETE FROM professors")
    cursor.execute("DELETE FROM syllabus")
    cursor.execute("DELETE FROM pyqs")
    cursor.execute("DELETE FROM deadlines")
    cursor.execute("DELETE FROM locations")

    cursor.executemany('INSERT OR IGNORE INTO courses VALUES (?,?,?,?)', courses)
    cursor.executemany('INSERT OR IGNORE INTO professors VALUES (?,?,?,?,?)', professors)
    cursor.executemany('INSERT OR IGNORE INTO syllabus VALUES (?,?,?,?)', syllabi)
    cursor.executemany('INSERT OR IGNORE INTO pyqs VALUES (?,?,?,?)', pyqs)
    cursor.executemany('INSERT OR IGNORE INTO deadlines (title, description, due_date, status) VALUES (?,?,?,?)', deadlines)
    cursor.executemany('INSERT OR IGNORE INTO locations VALUES (?,?,?,?)', locations)

    # --- 4. ADD SPECIFIC HARDCODED DATA (For Demo Consistency) ---
    # Ensures your specific demo queries (Priya, Vikram, Web Sec) always work
    specific_profs = [
        ('Dr. Vikram Singh', 'vikram.singh@university.edu', 'Computer Science', 'D-401', 'Artificial Intelligence'),
        ('Prof. Priya Sharma', 'priya.sharma@university.edu', 'Computer Science', 'A-101', 'Data Structures'),
    ]
    cursor.executemany('INSERT OR IGNORE INTO professors VALUES (?,?,?,?,?)', specific_profs)

    specific_courses = [
        ('CSET365', 'Web Security', 3, 'Computer Science'),
        ('CSET301', 'AI & Machine Learning', 4, 'Computer Science'),
    ]
    cursor.executemany('INSERT OR IGNORE INTO courses VALUES (?,?,?,?)', specific_courses)
    
    specific_deadlines = [
        ('ML Assignment 1', 'Neural Networks', '2025-11-28', 'pending'),
        ('Web Security Project', 'Vuln Scan', '2025-10-20', 'completed')
    ]
    cursor.executemany('INSERT OR IGNORE INTO deadlines (title, description, due_date, status) VALUES (?,?,?,?)', specific_deadlines)

    conn.commit()
    
    # Verification
    total = 0
    for table in ['courses', 'professors', 'syllabus', 'pyqs', 'deadlines', 'locations']:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table.capitalize()}: {count} entries")
        total += count

    conn.close()
    print(f"\n✅ Success! Database 'university.db' populated with {total} total entries.")

if __name__ == '__main__':
    create_database()