Intelligent Online Examination Proctoring System
📌 Overview

Ensuring the security of online examinations is essential in modern e-learning.
This project introduces an intelligent proctoring system that combines facial recognition and machine learning for precise and continuous identity verification of exam participants.

The system uses:

Dlib’s ResNet model for facial detection & feature encoding.

HOG (Histogram of Oriented Gradients) + SVM for robust authentication.

Eye Blink Detection (using EAR) to prevent spoofing via photos or videos.

Multi-face detection to ensure only the registered participant is present.

Real-time re-verification throughout the exam.

JavaScript-based activity tracking to detect window switching or inactivity.

MySQL for storing facial encodings and exam data.

This framework enhances exam security by preventing impersonation, cheating, and unauthorized access.

✨ Features

Registration with Face Encoding
Captures and encodes the participant’s facial features using Dlib’s deep learning-based ResNet model.

Spoof Prevention via Eye Blink Detection
Uses Eye Aspect Ratio (EAR) to detect natural blinking patterns, ensuring the person is live.

Multi-Face Detection
Flags and alerts if more than one person is detected in the camera frame.

Continuous Identity Verification
Periodic checks compare real-time facial data with stored encodings to ensure the participant remains the same.

Exam Activity Monitoring
Tracks browser tab/window switching, loss of focus, or suspicious inactivity via JavaScript.

Secure Data Storage
Stores facial encodings and exam details in a MySQL database for quick and reliable verification.

🛠️ Tech Stack

Backend: Python (Flask)

Machine Learning: Dlib (ResNet, HOG, SVM)

Frontend: HTML, CSS, JavaScript

Database: MySQL
🎯 How It Works

User Registration

Captures facial features and stores encodings in MySQL.

Exam Start

System verifies face match before allowing access.

During Exam

Continuous face re-verification.

Eye blink detection to ensure a live participant.

Multi-face detection to prevent unauthorized presence.

Browser activity tracking to detect cheating attempts.

Result

Scores calculated and displayed securely.

🔒 Security Measures

Prevents impersonation via real-time facial recognition.

Blocks photo/video spoofing using EAR-based blink detection.

Detects multiple people in camera view.

Alerts for suspicious activities like window switching.
Libraries: OpenCV, NumPy, Dlib

Security Features: EAR-based blink detection, multi-face verification, activity tracking
