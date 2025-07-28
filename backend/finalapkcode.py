import glob
import json
import os
import random
import smtplib
import threading
import time
from datetime import datetime, date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import base64
import cv2
import numpy as np
import requests
import asyncio
import edge_tts
import pygame
import io

from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
from fpdf import FPDF
from apscheduler.schedulers.background import BackgroundScheduler

# --- Configuration Constants ---
SAMPLES_PER_USER: int = 10
FRAME_REDUCE_FACTOR: float = 0.5
RECOGNITION_INTERVAL: int = 3 * 60  # 3 minutes cooldown

HAAR_CASCADE_PATH: str = "./haarcascade_frontalface_default.xml"

# Google Sheet URLs and form URLs (update with your actual URLs)
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/1ghgpG1z4ugXpu4cfOZjkPfPh-7oAQvOZhVRz2XHfot0/export?format=csv"
)
GOOGLE_FORM_VIEW_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"
)
GOOGLE_FORM_POST_URL: str = "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"

FORM_FIELDS: Dict[str, str] = {
    "name": "entry.935510406",
    "emp_id": "entry.886652582",
    "date": "entry.1160275796",
    "time": "entry.32017675",
}

EMAIL_ADDRESS: str = os.environ.get("FACEAPP_EMAIL", "faceapp0011@gmail.com")
EMAIL_PASSWORD: str = os.environ.get("FACEAPP_PASS", "ytup bjrd pupf tuuj")
SMTP_SERVER: str = "smtp.gmail.com"
SMTP_PORT: int = 587
ADMIN_EMAIL_ADDRESS: str = os.environ.get(
    "FACEAPP_ADMIN_EMAIL", "projects@archtechautomation.com"
)


def Logger(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def python_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _crop_and_resize_for_passport(
    cv_image: np.ndarray, target_size: Tuple[int, int] = (240, 320)
) -> np.ndarray:
    """Crop and resize an image to a target aspect ratio and size."""
    h, w = cv_image.shape[:2]
    target_width, target_height = target_size

    target_aspect_ratio = target_width / target_height
    current_aspect_ratio = w / h

    if current_aspect_ratio > target_aspect_ratio:
        new_width = int(h * target_aspect_ratio)
        x_start = (w - new_width) // 2
        cropped_image = cv_image[:, x_start : x_start + new_width]
    elif current_aspect_ratio < target_aspect_ratio:
        new_height = int(w / target_aspect_ratio)
        y_start = (h - new_height) // 2
        cropped_image = cv_image[y_start : y_start + new_height, :]
    else:
        cropped_image = cv_image

    return cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)


class ComplimentGenerator:
    def __init__(self):
        self.user_compliment_history: Dict[str, Dict[str, List[str]]] = {}  # emp_id -> history
        self.compliment_template = [
            "ThankYou!, {name}!",
            "{name}, your positivity is contagious!",
            "Wow, {name}, you're radiating confidence!",
            "{name}, your smile brightens the office!",
            "Great to see you, {name}—looking professional as always!",
            "{name}, you're making a great impression today!",
            "Impressive look, {name}. Keep it up!",
            "{name}, your energy sets a fantastic example!",
        ]

    def get_daily_compliment(self, emp_id: str, name: str) -> str:
        today = date.today().isoformat()
        if emp_id not in self.user_compliment_history:
            self.user_compliment_history[emp_id] = {"dates": [], "compliments": []}

        # Remove compliments from previous days
        idx_to_remove = [
            i
            for i, d in enumerate(self.user_compliment_history[emp_id]["dates"])
            if d != today
        ]
        for i in reversed(idx_to_remove):
            self.user_compliment_history[emp_id]["dates"].pop(i)
            self.user_compliment_history[emp_id]["compliments"].pop(i)

        # Exclude compliments already used today
        available = [
            c
            for c in self.compliment_template
            if c not in self.user_compliment_history[emp_id]["compliments"]
        ]

        if not available:
            available = self.compliment_template.copy()
            self.user_compliment_history[emp_id]["compliments"].clear()

        compliment = random.choice(available)
        self.user_compliment_history[emp_id]["dates"].append(today)
        self.user_compliment_history[emp_id]["compliments"].append(compliment)
        return compliment.format(name=name)


class EdgeTTSHelper:
    """Helper class for instant Edge-TTS with a slightly slower cute female voice"""

    def __init__(self):
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=256)
        pygame.mixer.init()
        self.selected_voice = "en-IN-NeerjaNeural"
        Logger(f"[INFO] EdgeTTS initialized with voice: {self.selected_voice}")

    def speak(self, text: str) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_speak(text))
            loop.close()
            Logger(f"[INFO] INSTANTLY spoke: {text}")
        except Exception as e:
            Logger(f"[ERROR] Instant TTS error: {e}")

    async def _async_speak(self, text: str) -> None:
        try:
            communicate = edge_tts.Communicate(text=text, voice=self.selected_voice, rate="+8%", pitch="+10Hz")
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if audio_data:
                audio_buffer = io.BytesIO(audio_data)
                pygame.mixer.music.load(audio_buffer, "mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                Logger("[INFO] Audio playing INSTANTLY")
        except Exception as e:
            Logger(f"[ERROR] Edge-TTS async speak error: {e}")


class FaceAppBackend:
    def __init__(self):
        self.known_faces_dir: str = str(Path("./known_faces"))
        ensure_dir(self.known_faces_dir)
        Logger(f"[INFO] Known faces directory set to: {self.known_faces_dir}")

        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            Logger(f"[WARN] Failed to load Haar cascade from '{HAAR_CASCADE_PATH}'. Attempting fallback.")
            fallback_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(fallback_path)
            if self.face_cascade.empty():
                raise RuntimeError("Cannot load Haar cascade classifier.")
            Logger(f"[INFO] Successfully loaded Haar cascade from fallback path: '{fallback_path}'.")

        self.recognizer = None
        self.label_map = {}
        self.last_seen_time: Dict[str, float] = {}
        self.daily_first_recognition: Dict[str, str] = {}  # emp_id -> date string
        self.otp_storage: Dict[str, str] = {}
        self.pending_names: Dict[str, Optional[str]] = {}
        self.user_emails: Dict[str, str] = {}
        self.daily_attendance_status: Dict[str, str] = {}
        self.last_recognized_info: Dict[str, Any] = {}
        self.capture_mode: bool = False
        self.capture_target_count: int = 0
        self.capture_collected_count: int = 0
        self.capture_name: Optional[str] = None
        self.capture_emp_id: Optional[str] = None
        self.capture_start_index: int = 0
        self.capture_lock = threading.Lock()
        self.tts_helper = EdgeTTSHelper()
        self.compliment_gen = ComplimentGenerator()
        self.user_gender: Dict[str, str] = self._load_user_genders()
        self.last_unknown_greeting_time: float = 0.0 # New attribute to track last unknown greeting

        self._train_recognizer_and_load_emails()
        self.daily_attendance_status = self._load_daily_attendance_status()

    def _load_user_genders(self) -> Dict[str, str]:
        gender_map = {}
        try:
            gender_file = Path(self.known_faces_dir) / "user_genders.json"
            if gender_file.is_file():
                with gender_file.open("r", encoding="utf-8") as f:
                    gender_map = json.load(f)
        except Exception as e:
            Logger(f"[WARN] Could not load user_genders.json: {e}")
        return gender_map

    def _save_user_genders(self) -> None:
        try:
            with (Path(self.known_faces_dir) / "user_genders.json").open("w", encoding="utf-8") as f:
                json.dump(self.user_gender, f, indent=2)
            Logger(f"[INFO] Saved user_genders.json with {len(self.user_gender)} entries.")
        except IOError as exc:
            Logger(f"[ERROR] Could not save user_genders.json: {exc}")

    def save_user_gender(self, emp_id: str, gender: str) -> None:
        self.user_gender[emp_id] = gender
        self._save_user_genders()

    def get_honorific(self, emp_id: str) -> str:
        gender = self.user_gender.get(emp_id, None)
        if gender == "female":
            return "ma'am"
        else:
            return "sir"

    def _train_recognizer_and_load_emails(self):
        self.recognizer, self.label_map = self._train_recognizer()
        self.user_emails = self._load_emails()

    def _train_recognizer(self):
        images: List[np.ndarray] = []
        labels: List[int] = []
        label_map: Dict[int, Tuple[str, str]] = {}
        label_id = 0

        ensure_dir(self.known_faces_dir)

        files = sorted(os.listdir(self.known_faces_dir))
        for file in files:
            if not file.lower().endswith((".jpg", ".png")):
                continue
            try:
                file_name_no_ext = file[:-4]
                parts = file_name_no_ext.split("_")
                if len(parts) < 3:
                    Logger(f"[WARN] Skipping unrecognized filename format: {file}")
                    continue
                name = " ".join(parts[:-2])
                emp_id = parts[-2].upper()
            except Exception as e:
                Logger(f"[WARN] Skipping unrecognized filename format: {file} due to error: {e}")
                continue

            img_path = Path(self.known_faces_dir) / file
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                Logger(f"[WARN] Could not read image: {img_path}")
                continue

            img_resized = cv2.resize(img_gray, (200, 200))
            current_identity = (name.lower(), emp_id)

            if current_identity not in label_map.values():
                label_map[label_id] = current_identity
                labels.append(label_id)
                label_id += 1
            else:
                existing_label_id = next(k for k, v in label_map.items() if v == current_identity)
                labels.append(existing_label_id)

            images.append(img_resized)

        recognizer = None
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except Exception as e:
            Logger(f"[ERROR] OpenCV LBPHFaceRecognizer_create failed: {e}")
            return None, label_map

        if images:
            try:
                recognizer.train(images, np.array(labels))
                Logger(f"[INFO] Trained recognizer on {len(images)} images across {len(label_map)} identities.")
            except cv2.error as e:
                Logger(f"[ERROR] OpenCV training error: {e}")
                recognizer = None
        else:
            Logger("[INFO] No images found – recognizer disabled.")
            recognizer = None

        return recognizer, label_map

    def _load_emails(self) -> Dict[str, str]:
        emails_file = Path(self.known_faces_dir) / "user_emails.json"
        if emails_file.is_file():
            try:
                with emails_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                Logger(f"[ERROR] Could not read user_emails.json: {exc}")
        return {}

    def _save_email(self, emp_id: str, email: str) -> None:
        self.user_emails[emp_id] = email
        try:
            with (Path(self.known_faces_dir) / "user_emails.json").open("w", encoding="utf-8") as f:
                json.dump(self.user_emails, f, indent=2)
            Logger(f"[INFO] Saved user_emails.json with {len(self.user_emails)} entries.")
        except IOError as exc:
            Logger(f"[ERROR] Could not save user_emails.json: {exc}")

    def _load_daily_attendance_status(self) -> Dict[str, str]:
        attendance_file = Path(self.known_faces_dir) / "daily_attendance.json"
        if attendance_file.is_file():
            try:
                with attendance_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                Logger(f"[ERROR] Could not read daily_attendance.json: {exc}")
        return {}

    def _save_daily_attendance_status(self) -> None:
        try:
            with (Path(self.known_faces_dir) / "daily_attendance.json").open("w", encoding="utf-8") as f:
                json.dump(self.daily_attendance_status, f, indent=2)
            Logger(f"[INFO] Saved daily_attendance.json with {len(self.daily_attendance_status)} entries.")
        except IOError as exc:
            Logger(f"[ERROR] Could not save daily_attendance.json: {exc}")

    def _generate_otp(self) -> str:
        return str(random.randint(100000, 999999))

    def _send_email(
        self,
        recipient_email: str,
        subject: str,
        body_html: str,
        image_data: Optional[bytes] = None,
        image_cid: Optional[str] = None,
        pdf_data: Optional[bytes] = None,
        pdf_filename: Optional[str] = None,
    ) -> bool:
        msg = MIMEMultipart("related")
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body_html, "html"))

        if image_data and image_cid:
            image = MIMEImage(image_data, "jpeg")
            image.add_header("Content-ID", f"<{image_cid}>")
            image.add_header("Content-Disposition", "inline", filename="face.jpg")
            msg.attach(image)

        if pdf_data and pdf_filename:
            pdf_part = MIMEApplication(pdf_data, _subtype="pdf")
            pdf_part.add_header("Content-Disposition", "attachment", filename=pdf_filename)
            msg.attach(pdf_part)

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            Logger(f"[INFO] Email successfully sent to {recipient_email}")
            return True
        except Exception as exc:
            Logger(f"[ERROR] SMTP email send error: {exc}")
            return False

    def _send_otp_email(self, email: str, otp: str, name: str, emp_id: str, is_admin_email: bool = False) -> bool:
        if is_admin_email:
            subject = f"FaceApp Notification: Person Details - {name.title()} ({emp_id})"
            body_html = f"""
                Name: {name.title()}<br/>
                Employee ID: {emp_id}<br/>
                This is an admin notification for OTP generation.
            """
        else:
            subject = f"FaceApp: Your OTP for {name.title()} ({emp_id})"
            body_html = f"""
                Employee ID: {emp_id}<br/>
                Email: {email}<br/>
                Generated OTP: {otp}<br/><br/>
                Dear {name.title()},<br/>
                Your OTP is <b>{otp}</b>. It is valid for 10 minutes.<br/>
                Please use this OTP to proceed with your photo update or registration.
            """

        return self._send_email(email, subject, body_html)

    def _send_attendance_email(
        self,
        email: str,
        name: str,
        emp_id: str,
        detection_time: str,
        email_type: str,
        face_image_b64: str,
    ) -> None:
        current_date_display = datetime.now().strftime("%B %d, %Y")
        image_data = base64.b64decode(face_image_b64)
        image_cid = "face_detection_image"
        image_html = f'<br/><img src="cid:{image_cid}" alt="Face Image"/>'

        compliment = self.compliment_gen.get_daily_compliment(emp_id, name)

        if email_type == "in":
            subject = f"FaceApp Attendance: In-Time Recorded for {name.title()} ({emp_id})"
            body_html = f"""
                Your attendance has been recorded successfully.{image_html}<br/>
                Date: {current_date_display}<br/>
                In-Time: {detection_time}<br/>
                Compliment: {compliment}<br/>
                Thank you for being part of our team!
            """
        elif email_type == "out":
            subject = f"FaceApp Attendance: Out-Time Recorded for {name.title()} ({emp_id})"
            body_html = f"""
                Your out-time has been recorded successfully.{image_html}<br/>
                Date: {current_date_display}<br/>
                Out-Time: {detection_time}<br/>
                Thank you for your support today. Have a great evening!
            """
        else:
            Logger(f"[ERROR] Invalid email_type '{email_type}'")
            return

        sent = self._send_email(email, subject, body_html, image_data, image_cid)
        if sent:
            Logger(f"[INFO] Attendance email sent to {email} ({email_type})")
        else:
            Logger(f"[ERROR] Failed to send attendance email to {email}")

    def _submit_to_google_form(self, name: str, emp_id: str) -> None:
        payload = {
            FORM_FIELDS["name"]: name.title(),
            FORM_FIELDS["emp_id"]: emp_id,
            FORM_FIELDS["date"]: datetime.now().strftime("%d/%m/%Y"),
            FORM_FIELDS["time"]: datetime.now().strftime("%H:%M:%S"),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (FaceApp Attendance Bot)",
            "Referer": GOOGLE_FORM_VIEW_URL,
        }
        try:
            with requests.Session() as session:
                resp = session.post(GOOGLE_FORM_POST_URL, data=payload, headers=headers, timeout=10)
                if resp.status_code in (200, 302):
                    Logger("[INFO] Attendance submitted successfully to Google Form.")
                else:
                    Logger(f"[WARN] Google Form submission returned status {resp.status_code}")
        except requests.exceptions.RequestException as exc:
            Logger(f"[ERROR] Google Form submission error: {exc}")

    def _handle_successful_recognition(self, name: str, emp_id: str, face_roi_color: np.ndarray) -> None:
        Logger(f"[INFO] Handling attendance for {name} ({emp_id})")
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time_str = datetime.now().strftime("%H:%M:%S")
        user_email = self.user_emails.get(emp_id)
        honorific = self.get_honorific(emp_id)
        processed_face_image = _crop_and_resize_for_passport(face_roi_color, (240, 320))
        _, buffer = cv2.imencode(".jpg", processed_face_image)
        face_image_b64 = base64.b64encode(buffer).decode("utf-8")

        first_time_today = self.daily_first_recognition.get(emp_id, None)
        is_in_time = first_time_today != current_date

        if is_in_time:
            self.daily_first_recognition[emp_id] = current_date

        greetings = [
            f"Oho! Good morning {name} {honorific}",
            f"Welcome, {name} {honorific}",
            f"All set, {name} {honorific}?",
            f"Nice to see you, {name} {honorific}",
            f"Hello, {name} {honorific}",
            f"Hey!, {name} {honorific}",
            f"Morning, {name} {honorific}",
            f"Ready to go, {name} {honorific}?",
            f"Good to have you, {name} {honorific}",
            f"Checked in, {name} {honorific}",
        ]

        farewells = [
            f"Bye bye {name} {honorific}",
            f"Take care, {name} {honorific}",
            f"See you, {name} {honorific}",
            f"Goodbye, {name} {honorific}",
            f"See you soon, {name} {honorific}",
            f"Catch you later, {name} {honorific}",
            f"All done, {name} {honorific}",
            f"Peace out, {name} {honorific}",
            f"Till next time, {name} {honorific}",
            f"Logging out, {name} {honorific}",
        ]

        greeting_text = random.choice(greetings if is_in_time else farewells)
        threading.Thread(target=self.tts_helper.speak, args=(greeting_text,), daemon=True).start()
        Logger(f"[INFO] Played greeting sound: {greeting_text}")

        if user_email:
            if is_in_time:
                self.daily_attendance_status[emp_id] = current_date
                self._save_daily_attendance_status()
                threading.Thread(
                    target=self._send_attendance_email,
                    args=(user_email, name, emp_id, current_time_str, "in", face_image_b64),
                    daemon=True,
                ).start()
            else:
                threading.Thread(
                    target=self._send_attendance_email,
                    args=(user_email, name, emp_id, current_time_str, "out", face_image_b64),
                    daemon=True,
                ).start()
        else:
            Logger(f"[WARN] No email found for {name} ({emp_id})")

        threading.Thread(target=self._submit_to_google_form, args=(name, emp_id), daemon=True).start()

        self.last_recognized_info = {
            "name": name.title(),
            "emp_id": emp_id,
            "time": current_time_str,
            "image": face_image_b64,
            "greeting": greeting_text,
        }

    def process_frame(self, frame_data_b64: str) -> Dict[str, Any]:
        try:
            if "," in frame_data_b64:
                frame_data_b64 = frame_data_b64.split(",")[1]

            nparr = np.frombuffer(base64.b64decode(frame_data_b64), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"status": "error", "message": "Invalid image data"}

            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (int(w * FRAME_REDUCE_FACTOR), int(h * FRAME_REDUCE_FACTOR)))
            gray_small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
            results = []

            # Track if any known face was detected in this frame
            known_face_detected_in_frame = False

            for (x, y, w_s, h_s) in faces:
                x_full, y_full, w_full, h_full = [int(v / FRAME_REDUCE_FACTOR) for v in (x, y, w_s, h_s)]
                expansion_factor = 1.8
                exp_w = int(w_full * expansion_factor)
                exp_h = int(h_full * expansion_factor)
                center_x = x_full + w_full // 2
                center_y = y_full + h_full // 2
                exp_x = max(0, center_x - exp_w // 2)
                exp_y = max(0, center_y - exp_h // 2)
                frame_h, frame_w = frame.shape[:2]

                exp_x = min(exp_x, frame_w - exp_w)
                exp_y = min(exp_y, frame_h - exp_h)
                exp_w = min(exp_w, frame_w - exp_x)
                exp_h = min(exp_h, frame_h - exp_y)

                if exp_w <= 0 or exp_h <= 0:
                    continue

                color_face_roi = frame[exp_y : exp_y + exp_h, exp_x : exp_x + exp_w].copy()
                grayscale_face_roi = cv2.cvtColor(color_face_roi, cv2.COLOR_BGR2GRAY)

                name = "unknown"
                emp_id = ""
                conf = 1000

                if self.recognizer:
                    try:
                        label, conf = self.recognizer.predict(grayscale_face_roi)
                        name, emp_id = self.label_map.get(label, ("unknown", ""))
                    except Exception as e:
                        Logger(f"[ERROR] Recognizer prediction failed: {e}")

                face_info = {
                    "box": [x_full, y_full, w_full, h_full],
                    "name": name.title(),
                    "emp_id": emp_id,
                    "confidence": float(conf),
                    "status": "unknown",
                }

                # Handle capture mode
                if self.capture_mode:
                    with self.capture_lock:
                        if self.capture_collected_count < self.capture_target_count:
                            face_img_resized = cv2.resize(grayscale_face_roi, (200, 200))
                            filename = f"{self.capture_name}_{self.capture_emp_id}_{self.capture_start_index + self.capture_collected_count:03d}.jpg"
                            cv2.imwrite(str(Path(self.known_faces_dir) / filename), face_img_resized)
                            self.capture_collected_count += 1
                            Logger(f"[INFO] Captured sample {self.capture_collected_count}/{self.capture_target_count}")
                            face_info["capture_progress"] = f"{self.capture_collected_count}/{self.capture_target_count}"
                            face_info["status"] = "capturing"
                            time.sleep(0.5)  # Delay to prevent camera auto-off

                        if self.capture_collected_count >= self.capture_target_count:
                            Logger("[INFO] Capture complete – retraining recognizer.")
                            self.capture_mode = False
                            threading.Thread(target=self._retrain_after_capture, daemon=True).start()
                            threading.Thread(target=self.tts_helper.speak, args=("Registration process is done",), daemon=True).start()
                            face_info["status"] = "capture_complete"
                        results.append(face_info)
                        continue

                if conf < 60: # Threshold for known faces
                    known_face_detected_in_frame = True
                    now = time.time()
                    last_seen = self.last_seen_time.get(emp_id, 0)
                    if now - last_seen > RECOGNITION_INTERVAL:
                        self.last_seen_time[emp_id] = now
                        face_info["status"] = "recognized_new"
                        threading.Thread(
                            target=self._handle_successful_recognition,
                            args=(name, emp_id, color_face_roi),
                            daemon=True,
                        ).start()
                    else:
                        face_info["status"] = "recognized_recent"
                else:
                    # Face is 'unknown' or confidence is too low to be recognized
                    face_info["status"] = "unknown"
                    
                results.append(face_info)
            
            # If no known face was detected and there are unknown faces, greet them
            # Changed 'if faces:' to 'if len(faces) > 0:' to resolve the ambiguity error
            if not known_face_detected_in_frame and len(faces) > 0: 
                now = time.time()
                # Cooldown for unknown greetings to avoid spamming
                if now - self.last_unknown_greeting_time > RECOGNITION_INTERVAL: 
                    self.last_unknown_greeting_time = now
                    greeting_unknown_text = "Hello! Welcome To Arc htech, I am Nova voice assistant, how can I help you?"
                    threading.Thread(target=self.tts_helper.speak, args=(greeting_unknown_text,), daemon=True).start()
                    Logger(f"[INFO] Played greeting sound for unknown person: {greeting_unknown_text}")

            return {"status": "success", "faces": results}
        except Exception as e:
            Logger(f"[ERROR] Error processing frame: {e}")
            return {"status": "error", "message": str(e)}

    def _retrain_after_capture(self):
        self.recognizer, self.label_map = self._train_recognizer()
        Logger("[INFO] Recognizer retraining finished.")

    def start_capture_samples(self, name: str, emp_id: str, updating=False, sample_count=None) -> Dict[str, Any]:
        with self.capture_lock:
            if self.capture_mode:
                return {"status": "error", "message": "Already in capture mode."}

            resolved_name = name

            if updating and not resolved_name:
                resolved_name = next(
                    (nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id),
                    None,
                )

            if resolved_name is None:
                return {"status": "error", "message": "No existing face found for this ID."}
            elif not resolved_name:
                return {"status": "error", "message": "Name is required for new registration."}

            self.capture_name = resolved_name
            self.capture_emp_id = emp_id
            self.capture_target_count = sample_count if sample_count is not None else 5
            self.capture_collected_count = 0

            pattern = str(Path(self.known_faces_dir) / f"{self.capture_name}_{self.capture_emp_id}_*.jpg")
            existing_files = glob.glob(pattern)
            self.capture_start_index = len(existing_files)

            self.capture_mode = True
            Logger(f"[INFO] Starting sample capture for {emp_id} – target {self.capture_target_count} faces")
            return {"status": "success", "message": "Capture mode initiated."}

    def stop_capture_samples(self) -> Dict[str, Any]:
        with self.capture_lock:
            if not self.capture_mode:
                return {"status": "error", "message": "Not in capture mode."}
            self.capture_mode = False
            Logger("[INFO] Sample capture stopped.")
            return {"status": "success", "message": "Capture mode stopped."}

    def get_user_email(self, emp_id: str) -> Dict[str, Any]:
        email = self.user_emails.get(emp_id)
        name = next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), None)
        return {"status": "success", "email": email, "name": name}

    def send_otp_flow(self, emp_id: str, email: str, name: Optional[str] = None) -> Dict[str, Any]:
        resolved_name = (
            name or next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), "Unknown User")
        )

        otp = self._generate_otp()
        self.otp_storage[emp_id] = otp
        self.pending_names[emp_id] = resolved_name

        def _send_thread():
            user_mail_ok = self._send_otp_email(email, otp, resolved_name, emp_id, False)
            admin_mail_ok = self._send_otp_email(ADMIN_EMAIL_ADDRESS, otp, resolved_name, emp_id, True)
            if not user_mail_ok:
                Logger(f"[WARN] Failed to send OTP email to user {email}")
            if not admin_mail_ok:
                Logger(f"[WARN] Failed to send admin notification email")

        threading.Thread(target=_send_thread, daemon=True).start()
        return {"status": "success", "message": "OTP sending initiated."}

    def verify_otp(self, emp_id: str, otp_entered: str) -> Dict[str, Any]:
        if self.otp_storage.get(emp_id) == otp_entered:
            del self.otp_storage[emp_id]
            return {"status": "success", "message": "OTP verified successfully."}
        else:
            return {"status": "error", "message": "Incorrect OTP."}

    def register_user_email(self, emp_id: str, email: str) -> Dict[str, Any]:
        self._save_email(emp_id, email)
        return {"status": "success", "message": "Email registered."}

    def get_last_recognized_info(self) -> Dict[str, Any]:
        info = self.last_recognized_info
        if info:
            self.last_recognized_info = {}
            return {"status": "success", "info": info}
        return {"status": "no_new_info"}

    def generate_and_send_monthly_reports(self):
        Logger("[INFO] Starting monthly attendance report generation and emailing...")
        try:
            # 1. Download the Google Sheet CSV
            resp = requests.get(GOOGLE_SHEET_CSV_URL, timeout=30)
            if resp.status_code != 200:
                Logger(f"[ERROR] Failed to download attendance sheet CSV: HTTP {resp.status_code}")
                return

            csv_data = resp.content.decode("utf-8")
            Logger("[INFO] CSV data downloaded successfully.")

            # 2. Read CSV into DataFrame
            df = pd.read_csv(io.StringIO(csv_data))
            Logger(f"[INFO] DataFrame loaded with {len(df)} rows.")

            # Expected columns - adapt as per your sheet's columns
            required_cols = ["Name", "Employee Id", "Date", "Time"]
            for col in required_cols:
                if col not in df.columns:
                    Logger(f"[ERROR] Attendance CSV missing required column: {col}. Available columns: {df.columns.tolist()}")
                    return

            # Convert 'Date' column to datetime. Explicitly setting format for robustness.
            # IMPORTANT: Adjust 'format' string here to match your Google Sheet's exact date format.
            # Common formats: "%d/%m/%Y" (DD/MM/YYYY), "%m/%d/%Y" (MM/DD/YYYY), "%Y-%m-%d" (YYYY-MM-DD)
            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
            Logger(f"[INFO] Dates after initial parsing (first 5 rows of Date column):\n{df['Date'].head()}")


            # Parse 'Time' column safely and convert to datetime.time, fill missing as None
            def safe_parse_time(t):
                try:
                    # Convert to string first to handle potential non-string types
                    # Assuming time format is HH:MM:SS
                    return pd.to_datetime(str(t), format="%H:%M:%S", errors="coerce").time()
                except Exception:
                    return None

            df["Time"] = df["Time"].apply(safe_parse_time)
            Logger(f"[INFO] Times after initial parsing (first 5 rows of Time column):\n{df['Time'].head()}")


            # Drop rows where 'Date' is NaT (invalid date) or 'Time' is None (invalid time)
            initial_rows = len(df)
            df = df.dropna(subset=["Date", "Time"])
            Logger(f"[INFO] DataFrame after dropping invalid dates/times: {len(df)} rows (dropped {initial_rows - len(df)} rows).")


            # 3. Determine previous month's date range
            today = datetime.now().date()
            first_day_this_month = today.replace(day=1)
            last_day_last_month = first_day_this_month - timedelta(days=1)
            first_day_last_month = last_day_last_month.replace(day=1)

            df_prev_month = df[
                (df["Date"].dt.date >= first_day_last_month) & (df["Date"].dt.date <= last_day_last_month)
            ].copy()
            Logger(f"[INFO] Filtered previous month's data: {len(df_prev_month)} rows.")

            if df_prev_month.empty:
                Logger("[WARN] No attendance data found for the previous month. Skipping report generation.")
                return

            # Sort by 'Employee Id', 'Date', 'Time' for correct min/max aggregation
            df_prev_month = df_prev_month.sort_values(by=["Employee Id", "Date", "Time"])
            Logger("[INFO] Previous month's data sorted.")


            # Prepare attendance summary: emp_id -> date -> in_time, out_time
            attendance_summary = df_prev_month.groupby(['Employee Id', df_prev_month['Date'].dt.date])['Time'].agg(
                in_time=('min'),
                out_time=('max')
            ).reset_index()

            attendance_summary = attendance_summary.rename(columns={'Date': 'AttendanceDate'})
            attendance_summary['Employee Id'] = attendance_summary['Employee Id'].astype(str)
            Logger(f"[INFO] Attendance summary created with {len(attendance_summary)} entries.")


            # Create full date range for last month (for consistent reporting, even if no attendance on a day)
            full_dates = pd.date_range(start=first_day_last_month, end=last_day_last_month).date

            # Send report to each employee
            all_emp_ids = attendance_summary['Employee Id'].unique()
            Logger(f"[INFO] Found {len(all_emp_ids)} unique employees for reporting.")

            for emp_id in all_emp_ids:
                # Try to get name
                name = next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), None)

                if not name:
                    # Try from DataFrame if not found in label_map
                    df_names = df_prev_month[df_prev_month["Employee Id"] == emp_id]["Name"].unique()
                    if len(df_names) > 0 and pd.notnull(df_names[0]):
                        name = df_names[0]
                    else:
                        name = "Unknown User"

                Logger(f"[INFO] Processing report for employee {emp_id} (Name: {name}, Email: {self.user_emails.get(emp_id, 'N/A')}).")

                email = self.user_emails.get(emp_id)
                if not email:
                    Logger(f"[WARN] No email found for {name} ({emp_id}), skipping report email.")
                    continue

                report_rows = []
                user_attendance_summary = attendance_summary[attendance_summary['Employee Id'] == emp_id]
                total_present_days = 0

                for single_date in full_dates:
                    daily_entry = user_attendance_summary[user_attendance_summary['AttendanceDate'] == single_date]
                    
                    in_time_str = ""
                    out_time_str = ""

                    if not daily_entry.empty:
                        total_present_days += 1
                        in_time_val = daily_entry['in_time'].iloc[0]
                        out_time_val = daily_entry['out_time'].iloc[0]
                        
                        if pd.notnull(in_time_val):
                            in_time_str = in_time_val.strftime('%H:%M:%S')
                        
                        # Set out_time only if it's different from in_time (implies at least two distinct entries)
                        if pd.notnull(out_time_val) and in_time_val != out_time_val:
                            out_time_str = out_time_val.strftime('%H:%M:%S')
                        
                    report_rows.append(
                        {
                            "Date": single_date.strftime("%Y-%m-%d"),
                            "In Time": in_time_str,
                            "Out Time": out_time_str,
                        }
                    )

                # Generate PDF
                pdf = FPDF()
                pdf.add_page()

                # Logos
                logo_left_path = Path(self.known_faces_dir) / "nextgen.png"
                logo_right_path = Path(self.known_faces_dir) / "logo.jpg"
                logo_w = 26
                logo_h = 11.87

                if logo_left_path.is_file():
                    pdf.image(str(logo_left_path), x=11, y=8, w=logo_w, h=logo_h)
                else:
                    Logger(f"[WARN] Left logo not found at {logo_left_path}")

                if logo_right_path.is_file():
                    pdf.image(str(logo_right_path), x=pdf.w - logo_w - 10, y=8, w=logo_w, h=logo_h)
                else:
                    Logger(f"[WARN] Right logo not found at {logo_right_path}")

                # Title and subheaders
                pdf.set_font("Arial", "B", 18)
                pdf.set_text_color(0, 0, 128)
                pdf.cell(0, 12, f"Attendance Report for {name}", ln=True, align="C")
                pdf.ln(4)
                pdf.set_font("Arial", "", 12)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 8, f"Employee ID: {emp_id}", ln=True, align="C")
                pdf.cell(0, 8, f"Period: {first_day_last_month.strftime('%B %Y')}", ln=True, align="C")
                pdf.ln(12)

                # Table header
                pdf.set_fill_color(200, 220, 255)
                pdf.set_font("Arial", "B", 12)
                col_width_date = 63.34
                col_width_in = 63.33
                col_width_out = 50

                pdf.cell(col_width_date, 10, "Date", border=1, fill=True, align="C")
                pdf.cell(col_width_in, 10, "In Time", border=1, fill=True, align="C")
                pdf.cell(col_width_out, 10, "Out Time", border=1, fill=True, align="C")
                pdf.ln()

                # Table content
                pdf.set_font("Arial", "", 12)
                for row in report_rows:
                    pdf.cell(col_width_date, 10, row["Date"], border=1)
                    pdf.cell(col_width_in, 10, row["In Time"], border=1)
                    pdf.cell(col_width_out, 10, row["Out Time"], border=1)
                    pdf.ln()

                pdf.ln(10)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Total Present Days: {total_present_days}", ln=True, align="L")

                pdf_output = pdf.output(dest="S").encode("latin1")
                Logger(f"[INFO] PDF generated for {name} ({emp_id}). Attempting to send email.")

                subject = f"Attendance Report for {name} - {first_day_last_month.strftime('%B %Y')}"
                body_html = f"""
                    Dear {name},<br/><br/>
                    Please find attached your attendance report for {first_day_last_month.strftime('%B %Y')}.<br/><br/>
                    Best regards,<br/>
                    FaceApp Attendance System
                """

                send_ok = self._send_email(
                    email,
                    subject,
                    body_html,
                    pdf_data=pdf_output,
                    pdf_filename=f"Attendance_Report_{first_day_last_month.strftime('%Y_%m')}.pdf",
                )
                if send_ok:
                    Logger(f"[INFO] Attendance report emailed successfully to {email}")
                else:
                    Logger(f"[ERROR] Failed to email attendance report to {email}")

        except Exception as e:
            Logger(f"[ERROR] Exception during monthly report generation: {e}")


# Initialize backend instance globally
backend_instance = FaceAppBackend()

scheduler = BackgroundScheduler()

# Schedule the first report to run immediately on startup
scheduler.add_job(
    backend_instance.generate_and_send_monthly_reports,
    "date",
    run_date=datetime.now(),
)

# Schedule subsequent reports for the 31st of every month at 1 AM
# This only triggers if a month has 31 days
scheduler.add_job(
    backend_instance.generate_and_send_monthly_reports,
    "cron",
    day="31",
    hour=1,
    minute=0,
)

scheduler.start()

app = Flask(__name__)
CORS(app)

face_app_backend = backend_instance


@app.route("/")
def index():
    return "FaceApp Backend with INSTANT greetings and voice prompt on registration running!"


@app.route("/process_frame", methods=["POST"])
def process_frame_endpoint():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"status": "error", "message": "No image data provided"}), 400

    frame_data_b64 = data["image"]
    if "," in frame_data_b64:
        frame_data_b64 = frame_data_b64.split(",")[1]

    result = face_app_backend.process_frame(frame_data_b64)
    return jsonify(result)


@app.route("/register_user", methods=["POST"])
def register_user_endpoint():
    data = request.json
    name = data.get("name")
    emp_id = data.get("emp_id")
    email = data.get("email")
    gender = data.get("gender")

    if not all([name, emp_id, email, gender]):
        return jsonify({"status": "error", "message": "Missing name, employee ID, email, or gender"}), 400

    if gender.lower() not in ("male", "female"):
        return jsonify({"status": "error", "message": "Invalid gender, must be 'male' or 'female'."}), 400

    if "@" not in email:
        return jsonify({"status": "error", "message": "Invalid email format"}), 400

    # Save email and gender persistently
    face_app_backend.register_user_email(emp_id, email)
    face_app_backend.save_user_gender(emp_id, gender.lower())

    def registration_tts_sequence():
        face_app_backend.tts_helper.speak("Hi I am Nova Voice assistant")
        time.sleep(0.3)
        face_app_backend.tts_helper.speak("I am clicking your photo")
        time.sleep(0.3)
        face_app_backend.tts_helper.speak("3 2 1")

    def tts_and_capture():
        registration_tts_sequence()
        face_app_backend.start_capture_samples(name, emp_id, updating=False, sample_count=5)

    threading.Thread(target=tts_and_capture, daemon=True).start()

    return jsonify({"status": "success", "message": "Registration process started, please follow voice instructions."})


@app.route("/get_user_email", methods=["POST"])
def get_user_email_endpoint():
    data = request.json
    emp_id = data.get("emp_id")

    if not emp_id:
        return jsonify({"status": "error", "message": "Missing employee ID"}), 400

    result = face_app_backend.get_user_email(emp_id)
    return jsonify(result)


@app.route("/send_otp", methods=["POST"])
def send_otp_endpoint():
    data = request.json
    emp_id = data.get("emp_id")
    email = data.get("email")
    name = data.get("name")

    if not all([emp_id, email]):
        return jsonify({"status": "error", "message": "Missing employee ID or email"}), 400

    result = face_app_backend.send_otp_flow(emp_id, email, name)
    return jsonify(result)


@app.route("/verify_otp", methods=["POST"])
def verify_otp_endpoint():
    data = request.json
    emp_id = data.get("emp_id")
    otp = data.get("otp")

    if not all([emp_id, otp]):
        return jsonify({"status": "error", "message": "Missing employee ID or OTP"}), 400

    result = face_app_backend.verify_otp(emp_id, otp)
    return jsonify(result)


@app.route("/start_update_capture", methods=["POST"])
def start_update_capture_endpoint():
    data = request.json
    name = data.get("name")
    emp_id = data.get("emp_id")

    result = face_app_backend.start_capture_samples(name, emp_id, updating=True, sample_count=5)
    return jsonify(result)


@app.route("/get_last_recognized", methods=["GET"])
def get_last_recognized_endpoint():
    result = face_app_backend.get_last_recognized_info()
    return jsonify(result)


if __name__ == "__main__":
    # Suppress Flask scheduler exits on reload
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
