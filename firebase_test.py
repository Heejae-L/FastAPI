import firebase_admin
from firebase_admin import credentials, storage

# 1. Firebase Admin SDK 초기화
cred = credentials.Certificate("/job-interview-46c64-firebase-adminsdk-uh8i0-a039e83be9.json")  # 서비스 계정 키 JSON 파일 경로
firebase_admin.initialize_app(cred, {
    'storageBucket': 'job-interview-46c64.firebasestorage.app'  # Firebase Storage 버킷 URL
})

bucket = storage.bucket()

# 2. 이미지 업로드 함수
def upload_image(local_file_path, cloud_file_name):
    blob = bucket.blob(cloud_file_name)
    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to {cloud_file_name}.")

# 3. 이미지 다운로드 함수
def download_image(cloud_file_name, local_file_path):
    blob = bucket.blob(cloud_file_name)
    blob.download_to_filename(local_file_path)
    print(f"File {cloud_file_name} downloaded to {local_file_path}.")

# 테스트용 코드
if __name__ == "__main__":
    # 로컬 이미지 파일 경로
    local_image_path = "/Users/LeeHeejae/projects/mediapipe_pose/server_1207/test_image.png"  # 업로드할 파일
    cloud_image_name = "test_image.png"  # Firebase Storage에서 저장될 이름

    # 업로드
    upload_image(local_image_path, cloud_image_name)

    # 다운로드
    download_path = "test_image.png"  # 다운로드할 파일 경로
    download_image(cloud_image_name, download_path)
