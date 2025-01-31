import base64
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
import pickle
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# MTCNN 모델 로드 (얼굴 검출)
mtcnn = MTCNN(keep_all=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')

# InceptionResnetV1 모델 로드 (FaceNet)
model = InceptionResnetV1(pretrained='vggface2').eval()

# 얼굴 임베딩 함수
def get_embedding(model, face):
    face = face.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face)
    return embedding[0].cpu().numpy()

# 분류기를 데이터베이스에서 로드하는 함수
def load_model_from_db(db_path):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('SELECT model FROM model_storage')
        model_blob = c.fetchone()[0]
        
        # 모델 역직렬화
        model = pickle.loads(model_blob)
    
    return model

# 임계값 설정
threshold = 0.6

# 데이터베이스 경로 설정 (face 폴더의 face_embeddings.db 파일)
folder_path = os.path.join(os.path.expanduser("~"), 'Desktop/face')
db_path = os.path.join(folder_path, 'face_embeddings.db')

if not os.path.exists(db_path):
    raise FileNotFoundError(f"데이터베이스 경로를 찾을 수 없습니다: {db_path}")

# k-NN 분류기 로드
knn = load_model_from_db(db_path)
print("k-NN 분류기 로드 완료")

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(frame_rgb)

    responses = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue  # 얼굴 영역이 비어있으면 pass
            face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
            face = (torch.tensor(face).permute(2, 0, 1).float() / 255.0)  # 정규화

            embedding = get_embedding(model, face)

            # 학습된 k-NN 모델을 사용하여 예측
            prediction = knn.predict([embedding])
            label = prediction[0]

            # k-NN에서 가장 가까운 학습된 임베딩과의 거리 계산
            distances, _ = knn.kneighbors([embedding], n_neighbors=1, return_distance=True)
            closest_dist = distances[0][0]

            # 가장 가까운 거리가 임계값보다 크면 'Unknown'으로 설정
            if closest_dist > threshold:
                label = 'Unknown'
            
            # 정확도 계산
            accuracy = (1 - closest_dist) * 100

            responses.append({
                'label': label,
                'box': [x1, y1, x2, y2],
                'color': 'green' if closest_dist <= threshold else 'red',
                'accuracy': accuracy
            })
    
    if not responses:
        responses.append({'label': 'No face detected'})

    return responses

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data_image):
    print("Received image data")  # 디버깅 로그 추가
    # 이미지 데이터를 바이트 배열로 변환
    try:
        image_data = np.frombuffer(base64.b64decode(data_image), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        responses = process_frame(img)
        print(f"Responses: {responses}")  # 디버깅 로그 추가
        emit('response_back', {'faces': responses})
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True)
