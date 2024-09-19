import numpy as np
from prepare_dataset import prepare_dataset
from create_and_train_model import create_and_train_model

def main():
    json_path = './data_set/TL_07.족제비/A07_H20_G070_C_211204_2085_20S_000008.010.json'  # 실제 JSON 파일 경로로 변경
    img_dir = './data_set/TS_07.족제비'
    
    # 데이터 준비
    X_train, img_files, channels = prepare_dataset(json_path, img_dir)
    y_train = np.array([1] * len(X_train))  # 더미 레이블 (실제 레이블로 교체 필요)
    
    # input_shape 설정
    input_shape = (1920, 1080, channels)
    
    # 모델 생성 및 학습
    model = create_and_train_model(X_train, y_train, input_shape)
    
    # 모델 저장
    model.save('./saved_model/my_model')
    
    # 이미지 파일명 출력
    print("처리된 이미지 파일들:", img_files)

if __name__ == "__main__":
    main()
