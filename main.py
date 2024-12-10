from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
import tempfile

# 1. Firebase Admin SDK 초기화
cred = credentials.Certificate("/pose-detection-test-d0122-firebase-adminsdk-ly39q-5bf1f0c175.json")  # 서비스 계정 키 JSON 파일 경로
firebase_admin.initialize_app(cred, {
    'storageBucket': 'pose-detection-test-d0122.firebasestorage.app'  # Firebase Storage 버킷 URL
})

bucket = storage.bucket()

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

app = FastAPI()

# 요청 JSON 스키마 정의
class AnalyzeRequest(BaseModel):
    email: str
    id: str
    name: str
    pdfUrl: Optional[str]
    question1: str
    question2: str
    question3: str
    question4: str
    videoPath: str
    timestamp: str

def analyze_frame(frame):
    # 이미지를 RGB로 변환
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_image)

    # MediaPipe 결과 처리
    feedback = []
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        feedback.append(check_knee_position(landmarks))
        feedback.append(check_back_straightness(landmarks))
        feedback.append(check_head_tilt(landmarks))
        feedback.append(check_facing_forward(landmarks))
    return [f for f in feedback if f]

# 다리 벌어짐 계산 함수
def check_knee_position(landmarks):
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    
    knee_distance = np.abs(left_knee.x - right_knee.x)
    hip_width = np.abs(left_knee.y - hip.y)
    
    if knee_distance > hip_width * 1.2:
        return "Don't spread your knees too much."
    return None

# 허리 기울어짐 계산 함수
def check_back_straightness(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
    hip_slope = abs(left_hip.y - right_hip.y)
    
    if shoulder_slope > 0.05 or hip_slope > 0.05:
        return "Keep your back straight."
    return None

# 고개 기울어짐 계산 함수
def check_head_tilt(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    
    tilt = abs(left_ear.y - right_ear.y)
    if tilt > 0.01:
        return "Your head is tilted."
    return None

# 정면 여부 판단 함수
def check_facing_forward(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
    
    eye_balance = abs(left_eye.x - right_eye.x)
    if eye_balance < 0.1:
        return None
    return "Face forward."

@app.post("/analyze/")
async def analyze_json(data: AnalyzeRequest):
    # JSON 데이터 처리
    extracted_data = {
        "email": data.email,
        "id": data.id,
        "name": data.name,
        "pdfUrl": data.pdfUrl,
        "question1": data.question1,
        "question2": data.question2,
        "question3": data.question3,
        "question4": data.question4,
        "videoPath": data.videoPath,
        "timestamp": data.timestamp
    }

    # Firebase Storage에서 비디오 파일 다운로드
    bucket = storage.bucket()
    blob = bucket.blob(data.videoPath)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="Video file not found in Firebase Storage.")

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        video_path = temp_file.name

    # 비디오 분석
    cap = cv2.VideoCapture(video_path)
    feedbacks = []
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open the video file for processing.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feedback = analyze_frame(frame)
        feedbacks.extend(feedback)

    cap.release()

    # 분석 결과 반환
    return {"extracted_data": extracted_data, "feedback": feedbacks}
