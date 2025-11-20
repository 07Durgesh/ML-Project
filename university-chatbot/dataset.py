# dataset.py
# EXTENDED TRAINING SET — Matches your 12 professors + 25 courses + syllabus + PYQ + deadlines + locations

TRAINING_QUERIES = [

    # =============== SYLLABUS & COURSES (Matches 25 Courses) ===============
    ("syllabus for CSET301", "syllabus"),
    ("AI & Machine Learning syllabus", "syllabus"),
    ("topics in AI and ML", "syllabus"),
    ("what is covered in Data Structures and Algorithms", "syllabus"),
    ("course content of Operating Systems", "syllabus"),
    ("DBMS syllabus CSET304", "syllabus"),
    ("content for Web Security", "syllabus"),
    ("Linear Algebra syllabus", "syllabus"),
    ("Calculus II course content", "syllabus"),
    ("Quantum Mechanics topics", "syllabus"),
    ("Thermodynamics syllabus", "syllabus"),
    ("Signals and Systems topics", "syllabus"),
    ("Organic Chemistry I syllabus", "syllabus"),
    ("Marketing course modules", "syllabus"),
    ("Cognitive Science topics", "syllabus"),
    ("Machine Learning Lab content", "syllabus"),
    ("NLP course syllabus", "syllabus"),
    ("Computer Networks modules", "syllabus"),
    ("Software Engineering topics", "syllabus"),
    ("Advanced Algorithms syllabus", "syllabus"),
    ("Distributed Systems content", "syllabus"),
    ("Information Retrieval syllabus", "syllabus"),
    ("Probability & Statistics syllabus", "syllabus"),
    ("Human Computer Interaction topics", "syllabus"),
    ("Capstone Project CSET395 structure", "syllabus"),
    ("download syllabus", "syllabus"),
    ("show curriculum", "syllabus"),
    ("course outline", "syllabus"),


    # =============== PROFESSORS & CONTACTS (Matches 12 Profs) ===============
    ("who teaches AI & Machine Learning", "professor"),
    ("professor for Data Structures", "professor"),
    ("who teaches Operating Systems", "professor"),
    ("DBMS faculty", "professor"),
    ("who teaches Web Security", "professor"),

    ("Linear Algebra teacher", "professor"),
    ("Mathematics professor", "professor"),
    ("Quantum Mechanics faculty", "professor"),
    ("Thermodynamics professor", "professor"),
    ("Signals and Systems teacher", "professor"),
    ("Organic Chemistry professor", "professor"),

    ("Marketing teacher", "professor"),
    ("Cognitive Science professor", "professor"),
    ("Machine Learning Lab incharge", "professor"),
    ("NLP course faculty", "professor"),
    ("Networks teacher", "professor"),
    ("Software Engineering faculty", "professor"),
    ("Advanced Algorithms professor", "professor"),
    ("Distributed Systems faculty", "professor"),
    ("Information Retrieval teacher", "professor"),
    ("Probability & Statistics faculty", "professor"),
    ("HCI professor", "professor"),
    ("Capstone guide", "professor"),

    # direct names
    ("contact Dr. Vikram Singh", "professor"),
    ("office of Prof. Priya Sharma", "professor"),
    ("email of Dr. Rajesh Kumar", "professor"),
    ("where does Dr. Emily Johnson sit", "professor"),
    ("Robert Brown department", "professor"),

    ("faculty details", "professor"),
    ("meet the professors", "professor"),


    # =============== PYQs (Matches 25 Courses) ===============
    ("PYQ for CSET301", "pyq"),
    ("AI & ML previous papers", "pyq"),
    ("old papers of Data Structures", "pyq"),
    ("Operating Systems question papers", "pyq"),
    ("DBMS past year papers", "pyq"),
    ("Web Security old papers", "pyq"),
    ("Linear Algebra PYQ", "pyq"),
    ("Calculus previous year papers", "pyq"),
    ("Quantum Mechanics previous exam", "pyq"),
    ("Thermodynamics question paper", "pyq"),
    ("Signals and Systems PYQ", "pyq"),
    ("Organic Chemistry past papers", "pyq"),
    ("Marketing previous exams", "pyq"),
    ("Cognitive Science PYQ", "pyq"),
    ("Machine Learning Lab question paper", "pyq"),
    ("NLP previous paper", "pyq"),
    ("Networks past exams", "pyq"),
    ("Software Engineering PYQ", "pyq"),
    ("Advanced Algorithms papers", "pyq"),
    ("Distributed Systems question set", "pyq"),
    ("Information Retrieval old papers", "pyq"),
    ("Probability & Statistics PYQ", "pyq"),
    ("HCI past exams", "pyq"),
    ("Capstone Project report samples", "pyq"),
    ("download old exam papers", "pyq"),


    # =============== DEADLINES (Reading) ===============
    ("upcoming deadlines", "deadline"),
    ("what tasks are pending", "deadline"),
    ("is any work due", "deadline"),
    ("show all deadlines", "deadline"),
    ("what is due this week", "deadline"),
    ("homework left", "deadline"),

    # Passed history
    ("show completed work", "deadline_history"),
    ("past deadlines", "deadline_history"),
    ("task history", "deadline_history"),


    # =============== DEADLINES (Writing) ===============
    ("add deadline", "add_deadline_intent"),
    ("add new task", "add_deadline_intent"),
    ("remind me about submission", "add_deadline_intent"),
    ("add assignment due on", "add_deadline_intent"),
    ("add project deadline", "add_deadline_intent"),

    ("mark task as done", "mark_deadline_intent"),
    ("mark assignment completed", "mark_deadline_intent"),
    ("finish work update", "mark_deadline_intent"),
    ("mark project as completed", "mark_deadline_intent"),


    # =============== LOCATIONS (Matches DB) ===============
    ("where is the library", "location"),
    ("library floor", "location"),
    ("location of cafeteria", "location"),
    ("where is tech park", "location"),
    ("computer lab location", "location"),
    ("find auditorium", "location"),
    ("admission office location", "location"),


    # =============== GREETING ==================
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey", "greeting"),
    ("restart", "greeting"),
    ("good morning", "greeting"),
    ("who are you", "greeting"),
    ("help", "greeting"),
    ("what can you do", "greeting"),
    # ======== DIRECT SUBJECT → COURSE CODE LINKS (25 COURSES) ========
("AI", "syllabus"),
("Artificial Intelligence", "syllabus"),
("Machine Learning", "syllabus"),
("AI ML", "syllabus"),
("ML", "syllabus"),
("AI and ML", "syllabus"),
("CSET301", "syllabus"),

("Data Structures", "syllabus"),
("Algorithms", "syllabus"),
("DSA", "syllabus"),
("CSET302", "syllabus"),

("Operating Systems", "syllabus"),
("OS", "syllabus"),
("CSET303", "syllabus"),

("Database Systems", "syllabus"),
("DBMS", "syllabus"),
("CSET304", "syllabus"),

("Web Security", "syllabus"),
("Cyber Security", "syllabus"),
("CSET305", "syllabus"),

("Calculus II", "syllabus"),
("Advanced Calculus", "syllabus"),
("MATH201", "syllabus"),

("Linear Algebra", "syllabus"),
("Matrices", "syllabus"),
("MATH202", "syllabus"),

("Physics I", "syllabus"),
("Mechanics physics", "syllabus"),
("PHYS101", "syllabus"),

("Quantum Mechanics", "syllabus"),
("PHYS102", "syllabus"),

("Thermodynamics", "syllabus"),
("MECH201", "syllabus"),

("Signals and Systems", "syllabus"),
("EE201", "syllabus"),

("Organic Chemistry", "syllabus"),
("CHEM101", "syllabus"),

("Principles of Marketing", "syllabus"),
("Marketing", "syllabus"),
("BUS101", "syllabus"),

("Cognitive Science", "syllabus"),
("PSY101", "syllabus"),

("Machine Learning Lab", "syllabus"),
("ML Lab", "syllabus"),
("CS350", "syllabus"),

("Natural Language Processing", "syllabus"),
("NLP", "syllabus"),
("CS360", "syllabus"),

("Computer Networks", "syllabus"),
("Networks", "syllabus"),
("CS370", "syllabus"),

("Software Engineering", "syllabus"),
("SE", "syllabus"),
("CS380", "syllabus"),

("Advanced Algorithms", "syllabus"),
("CSET401", "syllabus"),

("Distributed Systems", "syllabus"),
("CSET402", "syllabus"),

("Information Retrieval", "syllabus"),
("IR", "syllabus"),
("CSET403", "syllabus"),

("Digital Electronics", "syllabus"),
("EE301", "syllabus"),

("Probability & Statistics", "syllabus"),
("Probability", "syllabus"),
("Statistics", "syllabus"),
("MATH301", "syllabus"),

("Human Computer Interaction", "syllabus"),
("HCI", "syllabus"),
("CS390", "syllabus"),

("Capstone Project", "syllabus"),
("Final year project", "syllabus"),
("CS395", "syllabus"),
("where is computer lab", "location"),
("where is computer lab 1", "location"),
("computer lab location", "location"),
("lab 1 location", "location"),
("find computer lab 1", "location"),
("passed deadlines", "deadline_history"),
("show passed deadlines", "deadline_history"),
("deadlines already over", "deadline_history"),
("old deadlines", "deadline_history"),

]
