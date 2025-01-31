import os
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
from tqdm import tqdm
from torchvision import transforms
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MTCNN 모델 로드 (얼굴 검출)
logger.info("MTCNN 모델 로드 중...")
mtcnn = MTCNN(keep_all=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')

# InceptionResnetV1 모델 로드 (FaceNet)
logger.info("InceptionResnetV1 모델 로드 중...")
model = InceptionResnetV1(pretrained='vggface2').eval()

# 얼굴 임베딩 함수
def get_embedding(model, face):
    face = face.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face)
    return embedding[0].cpu().numpy()

# 데이터 증강 함수
def augment_image(image):
    augmentations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])
    return augmentations(image)

# 이미지 로드 및 얼굴 임베딩 생성 함수
def process_image(image_path, person_name):
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"이미지를 읽을 수 없습니다: {image_path}")
        return None, None

    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except cv2.error:
        logger.warning(f"이미지를 변환할 수 없습니다: {image_path}")
        return None, None

    box, _ = mtcnn.detect(img_rgb)
    if box is None:
        return None, None

    x1, y1, x2, y2 = map(int, box[0])
    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None, None

    face = cv2.resize(face, (160, 160))
    face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
    embedding = get_embedding(model, face)
    
    # 원본 임베딩 반환
    yield embedding, person_name
    
    # 증강된 임베딩 반환
    for _ in range(5):  # 각 이미지를 5배 증강
        augmented_face = augment_image(face)
        augmented_embedding = get_embedding(model, augmented_face)
        yield augmented_embedding, person_name

# 데이터베이스 초기화
def init_db(db_path):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                label TEXT NOT NULL
            )
        ''')
        conn.commit()

# 임베딩 저장 함수
def save_embeddings_to_db(db_path, embeddings):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.executemany('''
            INSERT INTO embeddings (embedding, label) VALUES (?, ?)
        ''', embeddings)
        conn.commit()

# 데이터셋 경로 설정 (바탕화면의 face 폴더)
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop/face')
dataset_path = os.path.join(desktop_path, 'facesDataset')
db_path = os.path.join(desktop_path, 'face_embeddings.db')

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {dataset_path}")

logger.info(f"데이터셋 경로: {dataset_path}")
logger.info(f"데이터베이스 경로: {db_path}")

# 데이터베이스 초기화
init_db(db_path)

# 데이터 로드 및 임베딩 생성
logger.info("데이터 로드 및 임베딩 생성 중...")
persons = os.listdir(dataset_path)
embeddings = []
for person_name in tqdm(persons, desc="사람 처리 중", unit="명"):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)
    for image_name in tqdm(images, desc=f"{person_name}의 이미지 처리 중", unit="장", leave=False):
        image_path = os.path.join(person_path, image_name)
        for embedding, label in process_image(image_path, person_name):
            if embedding is not None and label is not None:
                embeddings.append((embedding.tobytes(), label))

        # 배치로 임베딩 저장
        if len(embeddings) > 100:
            save_embeddings_to_db(db_path, embeddings)
            embeddings = []

# 남아 있는 임베딩 저장
if embeddings:
    save_embeddings_to_db(db_path, embeddings)

logger.info("임베딩 생성 및 데이터베이스 저장 완료")