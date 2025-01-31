1. 프로젝트 소개

FaceNet 딥러닝 모델을 사용하여 실시간 얼굴 인식을 구현한 프로그램

2. 파일 설명
app.py : Flask 웹 어플리케이션, 실시간 얼굴인식을 처리하는 시스템 구현

database.py :  데이터베이스 클래스 파일

imbedding.py : FaceNet을 사용하여 얼굴 임베딩 생성, SQLite 데이터베이스에 저장

ModelTrain.py : k-NN 분류기 사용하여 얼굴 임베딩 학습 및 성능 평가
