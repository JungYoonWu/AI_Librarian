# routers/detection.py (이상치 제거 로직 통합 최종 버전)

from ultralytics import YOLO
import numpy as np
from pathlib import Path

from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

# 모델 로드 (루트 폴더에 있는 모델을 사용)
model = YOLO("best.pt")

def calculate_obb_iou(corners1, corners2):
    """ 두 OBB의 IoU(Intersection over Union)를 계산합니다. """
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area

def perform_nms(detections, iou_threshold):
    """ OBB에 대한 NMS(Non-Maximal Suppression)를 수행합니다. """
    if not detections:
        return []
    
    # 신뢰도(confidence) 순으로 정렬
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    kept_detections = []
    
    while detections:
        # 가장 신뢰도 높은 박스를 선택하고 리스트에 추가
        best_detection = detections.pop(0)
        kept_detections.append(best_detection)
        
        # 나머지 박스들과 IoU 비교하여 임계값 미만인 것만 남김
        detections = [
            det for det in detections
            if calculate_obb_iou(best_detection['corners'], det['corners']) < iou_threshold
        ]
        
    return kept_detections

def detect_polygons(img: np.ndarray) -> list:
    """
    YOLO OBB 모델로 객체를 탐지한 후,
    NMS와 DBSCAN 클러스터링으로 결과를 정제하여 최종 폴리곤 리스트를 반환합니다.
    """
    # --- 필터링 파라미터 (이 값들을 조정하여 성능을 튜닝할 수 있습니다) ---
    IOU_THRESHOLD = 0.4         # NMS 임계값. 겹치는 영역이 이 값 이상이면 중복으로 간주.
    DBSCAN_EPS = 150            # DBSCAN의 이웃으로 간주할 최대 거리 (픽셀).
    DBSCAN_MIN_SAMPLES = 2      # 군집을 이루기 위한 최소 박스 수. 이보다 적으면 이상치.

    # 1. 모델 추론 실행
    # (verbose=False로 설정하여 서버 로그를 깔끔하게 유지)
    results = model.predict(img, conf=0.25, verbose=False)

    if not results or not results[0].obb:
        return []

    result = results[0]
    
    # 2. 모든 탐지 결과를 후처리하기 편한 형태로 변환
    detections_before_filtering = []
    if hasattr(result, 'obb') and result.obb is not None and len(result.obb.data) > 0:
        # result.obb.xyxyxyxy를 사용하면 직접 코너 좌표를 계산할 필요 없이 바로 사용 가능
        all_corners = result.obb.xyxyxyxy.cpu().numpy()
        all_confs = result.obb.conf.cpu().numpy()

        for i in range(len(all_corners)):
            detection = {
                'corners': [tuple(p) for p in all_corners[i]], # [(x,y), (x,y), ..]
                'confidence': float(all_confs[i])
            }
            detections_before_filtering.append(detection)

    if not detections_before_filtering:
        return []
    print(f"[Detect] 필터링 전 탐지된 객체 수: {len(detections_before_filtering)}")

    # 3. NMS 적용 (겹치는 박스 제거)
    detections_after_nms = perform_nms(detections_before_filtering, IOU_THRESHOLD)
    print(f"[Detect] NMS 후 남은 객체 수: {len(detections_after_nms)}")

    # 4. DBSCAN 적용 (동떨어진 이상치 박스 제거)
    #    박스가 2개 이상일 때만 클러스터링이 의미가 있습니다.
    if len(detections_after_nms) > DBSCAN_MIN_SAMPLES:
        # 각 박스의 첫 번째 코너 좌표를 기준으로 클러스터링
        points_for_clustering = np.array([d['corners'][0] for d in detections_after_nms])
        
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points_for_clustering)
        labels = db.labels_
        
        # 레이블이 -1인 박스(DBSCAN이 이상치로 판단한 박스)를 제외
        final_detections = [
            det for det, label in zip(detections_after_nms, labels) if label != -1
        ]
        print(f"[Detect] 이상치 제거 후 최종 객체 수: {len(final_detections)}")
    else:
        # 박스 수가 적으면 이상치 탐색을 건너뜀
        final_detections = detections_after_nms
        print(f"[Detect] 객체 수가 적어 이상치 탐색을 건너뜁니다.")

    # 5. 최종 결과 포맷팅 (서버의 다른 로직이 요구하는 폴리곤 좌표 리스트로 변환)
    #    CSV 저장이나 시각화 로직은 서버에 불필요하므로 제거합니다.
    final_polygons = [det['corners'] for det in final_detections]
    
    return final_polygons
