from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import pymysql
import face_recognition
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
import time
import threading
import webbrowser

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Database connection details
DB_HOST = "localhost"
DB_USER = username
DB_PASSWORD = password
DB_NAME = "face"

def get_db_connection():
    try:
        db = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor  # Ensures dictionary output
        )
        return db
    except pymysql.MySQLError as e:
        print(f"Database Connection Error: {e}")  # Print error for debugging
        return None


# Load dlib detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file exists

# Face recognition parameters
TARGET_BOX_AREA = 40000  # Ideal face size
TOLERANCE = 10000  # Allowed variation in face size
BLINK_THRESHOLD = 0.25  # EAR threshold for blink detection
WAIT_TIME = 40
# Global variable to track authentication state
is_authenticated = False


def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) to detect blinking."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def capture_face_for_verification(prompt, blink_cooldown=2):
    """Captures a face for verification, including distance feedback, blink detection, and lighting adjustment."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access the camera.")
        return None, None

    print(prompt)
    encoding = None
    user_ready = False
    blink_detected = False
    blink_time = None
    start_time = time.time()  # Start timer

    # Initialize CLAHE (for adaptive brightness control)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from the camera.")
            break

        # Apply gray lighting adjustment
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = clahe.apply(gray_frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        allow_blink = False  # Default to False at each frame

        if not user_ready:
            cv2.putText(frame, "Press 'S' to Start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("[INFO] User started the process.")
                user_ready = True
            continue

        if face_locations:
            if len(face_locations) > 1:
                cv2.putText(frame, "Multiple faces detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    box_width = right - left
                    box_height = bottom - top
                    box_area = box_width * box_height

                    if abs(box_area - TARGET_BOX_AREA) <= TOLERANCE:
                        feedback = "Perfect distance! Blink to Capture."
                        allow_blink = True
                    elif box_area > TARGET_BOX_AREA:
                        feedback = "Too close! Move back."
                    else:
                        feedback = "Too far! Move closer."

                    cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0) if allow_blink else (0, 0, 255), 2)

                    if allow_blink:
                        faces = detector(gray_frame)
                        for face in faces:
                            landmarks = predictor(gray_frame, face)
                            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

                            left_ear = eye_aspect_ratio(left_eye)
                            right_ear = eye_aspect_ratio(right_eye)
                            avg_ear = (left_ear + right_ear) / 2.0

                            if avg_ear < BLINK_THRESHOLD:
                                if not blink_detected:
                                    blink_detected = True
                                    blink_time = time.time()
                                    print("[INFO] Blink detected! Capturing in", blink_cooldown, "seconds...")

        else:
            cv2.putText(frame, "No face detected. Adjust position.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

        # Handle capture after blink
        if blink_detected and blink_time and allow_blink:
            elapsed_time = time.time() - blink_time
            if elapsed_time >= blink_cooldown:
                print("[INFO] Capturing face after blink cooldown...")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    encoding = face_encodings[0]
                    cap.release()
                    cv2.destroyAllWindows()
                    return frame, encoding
                else:
                    print("[ERROR] Unable to encode the face. Try again.")

        # Exit after wait time
        if time.time() - start_time >= WAIT_TIME:
            print("[INFO] No blink detected within time limit. Exiting...")
            break

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Face capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return None, None

def monitor_face_automatically(stored_encoding, student_id, exam_id):
    """Monitor face automatically during the exam session with bounding boxes and continuous re-verification."""
    cap = cv2.VideoCapture(0)
    running = True
    last_verified_time = time.time()
    reverification_interval = 10  # Check every 10 seconds
    mismatch_count = 0  # Track sequential mismatches

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from the camera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 1:  # Only one face detected
            # Draw bounding box around the face
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                current_time = time.time()
                if current_time - last_verified_time >= reverification_interval:
                    last_verified_time = current_time
                    print("[INFO] Re-verifying face...")

                    verified = compare_face(face_encodings[0], stored_encoding)
                    if verified:
                        print("[INFO] Face verified successfully.")
                        mismatch_count = 0  # Reset mismatch counter
                    else:
                        mismatch_count += 1
                        print(f"[ALERT] Face mismatch detected! Count: {mismatch_count}")

                        if mismatch_count >= 3:  # Report only after 3 sequential mismatches
                            print("[WARNING] 3 consecutive mismatches! Reporting to database...")
                            report_mismatch(student_id, exam_id)
                            mismatch_count = 0  # Reset after reporting

        elif len(face_locations) > 1:
            print("[WARNING] Multiple faces detected! The exam will be locked.")


        # Display the frame with bounding box
        cv2.imshow("Monitoring during exam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Monitoring cancelled.")
            running = False

    cap.release()
    cv2.destroyAllWindows()


def report_mismatch(student_id, exam_id, action="Mismatch detected, flagged for review"):
    """Writes a mismatch report to the MySQL database with an action description."""
    try:
        # Connect to MySQL
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        cursor = conn.cursor()

        # Ensure the report table exists with an action column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS report (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id VARCHAR(20),
                exam_id VARCHAR(20),
                action VARCHAR(100),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert the mismatch report with action
        cursor.execute("INSERT INTO report (student_id, exam_id, action) VALUES (%s, %s, %s)",
                       (student_id, exam_id, action))
        conn.commit()
        print("[INFO] Mismatch report successfully logged with action.")

    except pymysql.MySQLError as e:
        print(f"[ERROR] Database error: {e}")
    finally:
        cursor.close()
        conn.close()

def compare_face(live_encoding, stored_encoding):
    """Compare the live face encoding with the stored encoding."""
    results = face_recognition.compare_faces([stored_encoding], live_encoding, tolerance=0.4)
    return results[0]  # Return True if the face matches, False otherwise

def capture_face(prompt, blink_cooldown=2):
    """Captures a face, ensures correct distance, detects blinks, and captures automatically."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access the camera.")
        return None, None

    print(prompt)
    encoding = None
    user_ready = False
    blink_detected = False
    blink_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from the camera.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # Step 1: Show "Press 'S' to Start" until user presses 'S'
        if not user_ready:
            cv2.putText(frame, "Press 'S' to Start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("[INFO] User started the process.")
                user_ready = True
            continue

        # Step 2: Check face distance
        if face_locations:
            if len(face_locations) > 1:
                cv2.putText(frame, "Multiple faces detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Calculate face distance
                    box_width = right - left
                    box_height = bottom - top
                    box_area = box_width * box_height

                    if abs(box_area - TARGET_BOX_AREA) <= TOLERANCE:
                        feedback = "Perfect distance! Blink to Capture."
                        allow_blink = True
                    elif box_area > TARGET_BOX_AREA:
                        feedback = "Too close! Move back."
                        allow_blink = False
                    else:
                        feedback = "Too far! Move closer."
                        allow_blink = False

                    cv2.putText(frame, feedback, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if allow_blink else (0, 0, 255), 2)

                    # Step 3: Detect blink only if distance is correct
                    if allow_blink:
                        faces = detector(gray_frame)
                        for face in faces:
                            landmarks = predictor(gray_frame, face)
                            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

                            left_ear = eye_aspect_ratio(left_eye)
                            right_ear = eye_aspect_ratio(right_eye)
                            avg_ear = (left_ear + right_ear) / 2.0

                            if avg_ear < BLINK_THRESHOLD:
                                if not blink_detected:
                                    blink_detected = True
                                    blink_time = time.time()
                                    print("[INFO] Blink detected! Capturing in", blink_cooldown, "seconds...")

        else:
            cv2.putText(frame, "No face detected. Adjust position.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Step 4: Capture after blink cooldown
        if blink_detected and blink_time and allow_blink:
            elapsed_time = time.time() - blink_time
            if elapsed_time >= blink_cooldown:
                print("[INFO] Capturing face after blink cooldown...")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    encoding = face_encodings[0]
                    cap.release()
                    cv2.destroyAllWindows()
                    return frame, encoding
                else:
                    print("[ERROR] Unable to encode the face. Try again.")

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Face capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return None, None


@app.route('/register', methods=['POST'])
def register():
    """Registers a face and saves it to the database."""
    name = request.form.get('name')  # Get name from form input
    student_id = request.form.get('studentid')  # Get student ID
    email = request.form.get('email')  # Get email
    exam_id = request.form.get('exam')  # Get exam ID

    if not name or not student_id or not email or not exam_id:
        return "❌ Error: All fields (Name, Student ID, Email, Exam) are required!", 400  # Ensure all fields are present

    print(f"[INFO] Registering face for {name} (ID: {student_id}) for Exam ID {exam_id}...")

    # Capture face encoding
    frame, encoding = capture_face(f"Look at the camera for registration, {name}...")
    if encoding is None:
        return "❌ Registration failed. No face detected! Please try again.", 400  # Return a failure message

    encoding_bytes = encoding.tobytes()  # Convert numpy array to bytes

    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password="", database=DB_NAME)

        with conn.cursor() as cursor:
            # Check if student already exists
            cursor.execute("SELECT student_id FROM Student WHERE student_id = %s", (student_id,))
            existing_student = cursor.fetchone()

            if existing_student:
                return f"⚠️ Error: Student ID {student_id} is already registered!", 400  # Prevent duplicate registration

            # Insert into Student table
            cursor.execute(
                "INSERT INTO Student (student_id, name, email, face_encoding) VALUES (%s, %s, %s, %s)",
                (student_id, name, email, encoding_bytes)
            )

            # Insert into Student_exam table (linking student with the exam)
            cursor.execute(
                "INSERT INTO Student_exam (student_id, exam_id) VALUES (%s, %s)",
                (student_id, exam_id)
            )

            conn.commit()
            return f" Student {name} (ID: {student_id}) registered successfully for Exam ID {exam_id}!"

    except pymysql.IntegrityError as err:
        print(f"[ERROR] Integrity constraint failed: {err}")
        return " Database error: Ivalid exam code", 400

    except pymysql.MySQLError as err:
        print(f"[ERROR] Database operation failed: {err}")
        return " Database error occurred!", 500

    finally:
        conn.close()  # Ensure database connection is closed properly

# ✅ Initialize failed attempts dictionary
failed_attempts = {}

@app.route('/verify', methods=['POST'])
def verify():
    """Verifies a registered face and starts the exam if successful."""
    global is_authenticated

    name_to_verify = request.form['name']
    exam_id = request.form['exam']
    student_id=request.form['studentid']
    threshold = 0.4

    # ✅ Ensure failed_attempts is properly initialized
    if name_to_verify not in failed_attempts:
        failed_attempts[name_to_verify] = 0

    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password="", database=DB_NAME)
        cursor = conn.cursor()

        cursor.execute("SELECT face_encoding FROM student WHERE name = %s", (name_to_verify,))
        result = cursor.fetchone()

        if not result:
            print(f"[ERROR] No encoding found for {name_to_verify}. Please register first.")
            return redirect(url_for('index'))

        stored_encoding = np.frombuffer(result[0], dtype=np.float64)
        print("[INFO] Starting verification process...")

        _, live_encoding = capture_face_for_verification(f"Look at the camera for verification of {name_to_verify}...")

        if live_encoding is None:
            print(f"[ERROR] Face verification failed for {name_to_verify} (no face detected)")
            return "❌ Face not detected. Try again."

        if face_recognition.compare_faces([stored_encoding], live_encoding, tolerance=threshold)[0]:
            is_authenticated = True
            print(f"[SUCCESS] {name_to_verify} verified!")

            # ✅ Reset failed attempts on success
            failed_attempts[name_to_verify] = 0

            # ✅ Redirect to exam page
            exam_url = url_for('exam', exam_id=exam_id,stud_id=student_id)
            threading.Thread(target=webbrowser.open, args=(f"http://localhost:5000{exam_url}",)).start()
            threading.Thread(target=monitor_face_automatically, args=(stored_encoding,exam_id,student_id)).start()

            return redirect(exam_url)
        else:
            failed_attempts[name_to_verify] += 1
            print(f"[WARNING] Authentication failed ({failed_attempts[name_to_verify]}/3 attempts)")

            if failed_attempts[name_to_verify] >= 1:
                print(f"[ALERT] 3 failed attempts for {name_to_verify}. Locking out...")
                lock_url = url_for('lock_page')
                threading.Thread(target=webbrowser.open, args=(f"http://localhost:5000{lock_url}",)).start() # ✅ Redirect to lock.html

            return "❌ Authentication Failed! Try again."

    except pymysql.MySQLError as err:
        print(f"[ERROR] Database operation failed: {err}")
        return "❌ Database error occurred.", 500  # Return a proper HTTP error code

    finally:
        conn.close()  # ✅ Ensure connection is closed

@app.route('/lock_page')
def lock_page():
    """Shows lockout page after 3 failed attempts."""
    return render_template("lock.html")  # ✅ Ensure lock.html exists in `templates/`


@app.route("/start_exam/<int:exam_id>")
def start_exam(exam_id):
    """Fetches and displays exam questions for the verified student."""
    if not is_authenticated:
        return redirect(url_for('index'))  # Ensure only verified users access

    student_id = session.get("student_id")  # Retrieve student ID from session
    if not student_id:
        return redirect(url_for('index'))  # Ensure only logged-in students can access

    db = pymysql.connect(host="localhost", user="root", password="", database="face")
    cursor = db.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT duration FROM exam WHERE exam_id = %s", (exam_id,))
    exam = cursor.fetchone()
    duration = exam["duration"] if exam else 10  # Default to 10 minutes

    cursor.execute("SELECT * FROM question WHERE exam_id = %s", (exam_id,))
    questions = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template("exam.html", questions=questions, duration=duration, exam_id=exam_id, student_id=student_id)


@app.route("/exam")
def exam():
    """Fetches and displays questions for the selected exam."""
    if not is_authenticated:
        return redirect(url_for('index'))

    exam_id = request.args.get('exam_id')  # ✅ Get exam_id from URL
    student_id = request.args.get('stud_id')  # ✅ Get student_id from URL

    if not exam_id:
        return "❌ Exam ID not found!"
    if not student_id:
        return "❌ Student ID not found!"

    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password="", database=DB_NAME)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # ✅ Fetch exam details
    cursor.execute("SELECT exam_name, duration FROM exam WHERE exam_id = %s", (exam_id,))
    exam = cursor.fetchone()

    if not exam:
        return "❌ Exam not found!"

    exam_name = exam["exam_name"]
    duration = exam["duration"]

    # ✅ Fetch questions and structure them properly
    cursor.execute("""
        SELECT question_id, question_text, option_A, option_B, option_C, option_D, correct_option 
        FROM question WHERE exam_id = %s
    """, (exam_id,))

    raw_questions = cursor.fetchall()

    questions = []
    for q in raw_questions:
        question_data = {
            "question_id": q["question_id"],
            "question_text": q["question_text"],
            "options": {
                "A": q["option_A"],
                "B": q["option_B"],
                "C": q["option_C"],
                "D": q["option_D"]
            },
            "correct_option": q["correct_option"]  # ✅ Include correct answer for examiner view
        }
        questions.append(question_data)
        print(questions)
    cursor.close()
    conn.close()

    # ✅ Pass student_id and formatted questions to exam.html
    return render_template("exam.html", questions=questions, duration=duration, exam_id=exam_id, exam_name=exam_name,
                           student_id=student_id)


@app.route('/')
def index():
    return render_template('index.html')  # Loads index.html from templates folder
def connect_db():
    """Establishes a database connection using PyMySQL."""
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor  # Enables dictionary-style results
    )

# Route for Exam Creation
@app.route("/create_exam", methods=["GET", "POST"])
def create_exam():
    if request.method == "POST":
        exam_id = request.form["exam_id"]  # Ensure exam_id is retrieved correctly
        exam_name = request.form["exam_name"]
        subject_name = request.form["subject_name"]
        duration = request.form["duration"]
        teacher_id = request.form["teacher_id"]  # Teacher enters their ID manually

        db = connect_db()
        cursor = db.cursor()
        try:
            cursor.execute(
                "INSERT INTO exam (exam_id, exam_name, subject_name, duration, teacher_id) VALUES (%s, %s, %s, %s, %s)",
                (exam_id, exam_name, subject_name, duration, teacher_id)
            )

            if cursor.rowcount > 0:  # ✅ Check if insertion was successful
                db.commit()
                flash("Exam created successfully!", "success")
            else:
                flash("Exam creation failed. Please try again.", "error")
        except pymysql.MySQLError as e:
            flash(f"Error: {str(e)}", "danger")
        finally:
            cursor.close()
            db.close()

        return redirect(url_for("add_questions", exam_id=exam_id))

    return render_template("create_exam.html")

@app.route("/add_questions/<int:exam_id>", methods=["GET", "POST"])
def add_questions(exam_id):
    if request.method == "POST":
        question_text = request.form["question_text"]
        option_A = request.form["option_A"]
        option_B = request.form["option_B"]
        option_C = request.form["option_C"]
        option_D = request.form["option_D"]
        correct_option = request.form["correct_option"]

        db = connect_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO Question (exam_id, question_text, option_A, option_B, option_C, option_D, correct_option) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (exam_id, question_text, option_A, option_B, option_C, option_D, correct_option)
        )
        db.commit()
        cursor.close()
        db.close()

        return "Question added successfully!"

    # Handle GET request (Show the form)
    return render_template("add_questions.html", exam_id=exam_id)

# Route for Teacher Registration
@app.route("/teacher", methods=["GET", "POST"])
def teacher():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]

        db = connect_db()
        cursor = db.cursor()

        try:
            cursor.execute("INSERT INTO teacher (name, email) VALUES (%s, %s)", (name, email))
            db.commit()
            teacher_id = cursor.lastrowid  # Get the newly inserted teacher's ID
            return redirect(url_for("create_exam", teacher_id=teacher_id))

        except pymysql.IntegrityError as error:
            flash(str(error))

        finally:
            cursor.close()
            db.close()

    return render_template("teacher.html")

@app.route('/student')
def student():
    return render_template('index2.html')


@app.route('/get-exams', methods=['GET'])
def get_exams():
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)  # Ensure output is a dictionary

        cursor.execute("SELECT exam_id, exam_name FROM exams")  # Replace 'exams' with your table name
        exams = cursor.fetchall()

        cursor.close()
        db.close()

        return jsonify({"exams": exams})  # Return data as a dictionary
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API route to fetch exams for a specific student
@app.route('/get-exams/<student_id>', methods=['GET'])
def get_student_exams(student_id):
    try:
        db = get_db_connection()
        cursor = db.cursor()  # Remove dictionary=True

        query = """
        SELECT e.exam_id, e.exam_name 
        FROM exam e 
        JOIN student_exam se ON e.exam_id = se.exam_id 
        WHERE se.student_id = %s
        """
        cursor.execute(query, (student_id,))
        exams = cursor.fetchall()

        cursor.close()
        db.close()

        return jsonify({"exams": exams})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    try:
        # ✅ Extract request data
        data = request.json
        student_id = data.get("student_id")
        exam_id = data.get("exam_id")
        answers = data.get("answers", {})

        app.logger.info(f"Received a request to submit exam.")
        app.logger.info(f"Raw request data: {data}")
        app.logger.info(f"Extracted - Student ID: {student_id}, Exam ID: {exam_id}, Answers: {answers}")

        if not student_id or not exam_id:
            return jsonify({"error": "Missing student_id or exam_id!"}), 400

        # ✅ Connect to the database
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                app.logger.info("Connected to database.")

                # ✅ Store student answers (if any)
                if answers:
                    for question_id, student_answer in answers.items():
                        app.logger.info(f"Storing answer - QID: {question_id}, Answer: {student_answer}")
                        sql = """
                        REPLACE INTO student_answers (student_id, exam_id, question_id, student_answer) 
                        VALUES (%s, %s, %s, %s)
                        """
                        cursor.execute(sql, (student_id, exam_id, question_id, student_answer))
                    conn.commit()  # ✅ Ensure data is saved

                # ✅ Fetch correct answers from database
                app.logger.info(f"Fetching correct answers for exam_id: {exam_id}")
                cursor.execute("SELECT question_id, correct_option FROM question WHERE exam_id = %s", (exam_id,))
                rows = cursor.fetchall()

                app.logger.info(f"Fetched rows: {rows}")  # ✅ Debugging step

                if not rows:
                    return jsonify({"error": "No questions found for this exam!"}), 400

                # ✅ Extract correct answers properly
                correct_answers = {str(row["question_id"]): row["correct_option"].strip().upper() for row in rows}

                app.logger.info(f"Correct Answers: {correct_answers}")

                # ✅ Calculate Score (even if answers are empty)
                total_questions = len(correct_answers)
                correct_count = sum(1 for qid, ans in answers.items() if correct_answers.get(str(qid)) == ans.upper())

                percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
                score = correct_count  # ✅ Score as number of correct answers

                app.logger.info(f"Total Questions: {total_questions}, Correct: {correct_count}, Score: {score}, Percentage: {percentage:.2f}%")

                # ✅ Store the final score (even if 0 answers were submitted)
                cursor.execute("""
                    INSERT INTO exam_results (student_id, exam_id, score, percentage) 
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE score = VALUES(score), percentage = VALUES(percentage)
                """, (student_id, exam_id, score, percentage))

                conn.commit()  # ✅ Save score to database

        # ✅ Return the response in your required format
        return jsonify({
            "message": "Exam submitted successfully!",
            "student_id": student_id,
            "exam_id": exam_id,
            "score": score,
            "total_questions": total_questions,
            "percentage": round(percentage, 2)
        }), 200

    except Exception as e:
        app.logger.error(f"Exam Submission Error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/exam_results')
def exam_results():
    student_id = request.args.get("student_id")
    exam_id = request.args.get("exam_id")
    score = request.args.get("score")  # These come as strings
    total_questions = request.args.get("total_questions")
    percentage = request.args.get("percentage")

    if not student_id or not exam_id:
        return render_template("exam_result.html", error="Missing student ID or exam ID")

    # ✅ Convert numerical values to integers/floats
    score = int(score) if score is not None else 0
    total_questions = int(total_questions) if total_questions is not None else 1
    percentage = float(percentage) if percentage is not None else 0.0

    app.logger.info(f"Fetched results: score={score}, total_questions={total_questions}, percentage={percentage}")

    return render_template("exam_result.html",
        student_id=student_id,
        exam_id=exam_id,
        score=score,
        total_questions=total_questions,
        percentage=percentage)

@app.route("/report_violation", methods=["POST"])
def report_violation():
    data = request.json
    student_id = data.get("student_id")
    exam_id = data.get("exam_id")
    violation_reason = data.get("violation_reason")

    if not student_id or not exam_id or not violation_reason:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert a new violation record
        cursor.execute(
            "INSERT INTO report (student_id, exam_id, action) VALUES (%s, %s, %s)",
            (student_id, exam_id, violation_reason)
        )
        conn.commit()

        return jsonify({"message": "Violation recorded"}), 200

    except pymysql.MySQLError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
