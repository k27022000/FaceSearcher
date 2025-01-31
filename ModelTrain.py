import os
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터베이스에서 임베딩 로드 함수
def load_embeddings_from_db(db_path):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('SELECT embedding, label FROM embeddings')
        rows = c.fetchall()
    
    embeddings = []
    labels = []
    for row in rows:
        embedding = np.frombuffer(row[0], dtype=np.float32)
        label = row[1]
        embeddings.append(embedding)
        labels.append(label)
    
    return np.array(embeddings), np.array(labels)

# 분류기를 데이터베이스에 저장하는 함수
def save_model_to_db(db_path, model):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS model_storage (id INTEGER PRIMARY KEY, model BLOB)''')
        
        # 모델 직렬화
        model_blob = pickle.dumps(model)
        
        # 모델 저장
        c.execute('DELETE FROM model_storage')  # 기존 모델 삭제
        c.execute('DELETE FROM model_storage')
        c.execute('DELETE FROM model_storage')
        c.execute('DELETE FROM model_storage')
        c.execute('DELETE FROM model_storage')
        c.execute('INSERT INTO model_storage (model) VALUES (?)', (model_blob,))
        conn.commit()
        
# 데이터베이스에서 분류기를 로드하는 함수
def load_model_from_db(db_path):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('SELECT model FROM model_storage')
        model_blob = c.fetchone()[0]
        
        # 모델 역직렬화
        model = pickle.loads(model_blob)
    
    return model

# 데이터베이스 경로 설정 (face 폴더의 face_embeddings.db 파일)
folder_path = os.path.join(os.path.expanduser("~"), 'Desktop/face')
db_path = os.path.join(folder_path, 'face_embeddings.db')

if not os.path.exists(db_path):
    raise FileNotFoundError(f"데이터베이스 경로를 찾을 수 없습니다: {db_path}")

logger.info(f"데이터베이스 경로: {db_path}")

# 데이터 로드
logger.info("데이터 로드 중...")
data, labels = load_embeddings_from_db(db_path)
logger.info("데이터 로드 완료")

# 데이터셋을 학습 데이터와 테스트 데이터로 분할
logger.info("데이터셋 분할 중...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
logger.info("데이터셋 분할 완료")

# k-NN 분류기 학습
logger.info("k-NN 분류기 학습 중...")
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
logger.info("k-NN 분류기 학습 완료")

# 분류기 저장
logger.info("k-NN 분류기 저장 중...")
save_model_to_db(db_path, knn)
logger.info("k-NN 분류기 저장 완료")

# 테스트 데이터로 얼굴 인식 성능 평가
logger.info("테스트 데이터로 얼굴 인식 성능 평가 중...")
y_pred = knn.predict(X_test)

# 정확도, 정밀도, 재현율 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

logger.info(f'Accuracy: {accuracy}')
logger.info(f'Precision: {precision}')
logger.info(f'Recall: {recall}')

# 혼동 행렬 계산 및 출력
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
