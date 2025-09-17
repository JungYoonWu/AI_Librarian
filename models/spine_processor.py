#models/spine_processor.py
import cv2, numpy as np, math
import uuid
import re

def load_polygon_labels_from_text(txt: str, img_shape):
    """
    CSV 파일에서 절대 픽셀 좌표(x1,y1,x2,y2,x3,y3,x4,y4)를
    직접 읽어 폴리곤 좌표 리스트로 반환한다.
    (원래의 정규화된 YOLO(c x, c y, w, h) 방식이 아님에 주의)
    """
    polys = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue

        parts = ln.split(',')
        # (1) 헤더(“class_id” 등) 건너뛰기
        if not parts[0].isdigit():
            continue

        # (2) 최소한 16개 칼럼(class_id, class_name, confidence, x_center, y_center, width, height, angle_deg, x1, y1, x2, y2, x3, y3, x4, y4)이 있어야 함
        if len(parts) < 16:
            continue

        try:
            # CSV 9~16번째 칼럼이 이미 “절대 픽셀 x1, y1, x2, y2, x3, y3, x4, y4”임
            x1 = int(float(parts[8]));  y1 = int(float(parts[9]))
            x2 = int(float(parts[10])); y2 = int(float(parts[11]))
            x3 = int(float(parts[12])); y3 = int(float(parts[13]))
            x4 = int(float(parts[14])); y4 = int(float(parts[15]))
        except (ValueError, IndexError):
            # 숫자 파싱 실패하거나 칼럼 부족 시 건너뜀
            continue

        # (3) 이미지 경계를 벗어나는 좌표가 있으면 건너뛰기 (선택 사항)
        h, w = img_shape[:2]
        coords = [x1, y1, x2, y2, x3, y3, x4, y4]
        if any(c < 0 for c in coords) or any(c >= max(w, h) for c in coords):
            continue

        polys.append([
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x4, y4)
        ])

    return polys

# ─────────────────────────────────────────────
# 1) 라벨 → 픽셀 polygon
# ─────────────────────────────────────────────
def load_polygon_labels(label_path: str, img_shape):
    h, w = img_shape[:2]
    polys = []
    with open(label_path) as f:
        for ln in f:
            vals = list(map(float, ln.strip().split()))
            if len(vals) < 9:
                continue                     # cls cx cy … x4 y4
            pts = [(int(vals[i] * w), int(vals[i + 1] * h))
                   for i in range(1, 9, 2)]
            polys.append(pts)
    return polys


# ─────────────────────────────────────────────
# 2) 시각화용 컨투어
# ─────────────────────────────────────────────
def draw_spine_contour(img, pts, color=(255, 0, 255), thick=2):
    vis = img.copy()
    cv2.polylines(vis, [np.int32(pts)], True, color, thick, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────
# 3) polygon → 시계방향 정렬
# ─────────────────────────────────────────────
def _order_clockwise(pts):
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


# ─────────────────────────────────────────────
# 4) 책등 투시 + 세로방향 회전
# ─────────────────────────────────────────────
def rectify_spine_roi(img, poly):
    """
    poly: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 형태의 polygon 절대 좌표
    """
    # ① 시계방향 정렬
    src = _order_clockwise(poly)

    # ② 너비와 높이를 계산
    W = np.linalg.norm(src[0] - src[1])
    H = np.linalg.norm(src[0] - src[3])

    # ③ 이상치 필터링: 음수, 너무 작은 값, 이미지 크기를 크게 벗어나는 값
    img_h, img_w = img.shape[:2]
    if W <= 0 or H <= 0:
        return None  # 너비 혹은 높이가 0 또는 음수라면 무시

    # (이미지 크기의 두 배를 넘으면 이상치로 간주)
    if W > img_w * 2 or H > img_h * 2:
        return None

    # ④ 투시변환 대상 좌표 준비
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)

    # ⑤ 변환 행렬과 warpPerspective
    try:
        M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
        warp = cv2.warpPerspective(img, M, (int(W), int(H)), flags=cv2.INTER_CUBIC)
    except cv2.error:
        return None

    # ⑥ 잠정 ROI 크기가 너무 크지 않은지 한 번 더 확인
    if warp.size == 0:
        return None

    # (너비·높이가 다시 매우 클 경우, 메모리 과부하 방지 차원에서 건너뜀)
    if warp.shape[1] > img_w * 2 or warp.shape[0] > img_h * 2:
        return None

    # ⑦ 가로가 세로보다 크면 세로 방향이 되도록 90도 회전
    if warp.shape[1] > warp.shape[0]:
        try:
            warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except cv2.error:
            return None

    # ⑧ (디버그용) ROI가 제대로 잘렸는지 확인용 이미지 저장
    cv2.imwrite(f"debug_roi_{uuid.uuid4().hex}.png", warp)
    return warp


# ─────────────────────────────────────────────
# 5) 흰 배경 여백 제거 (빈-배열 방어)
# ─────────────────────────────────────────────
def trim_whitespace(bgr, thresh=10):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    nz = cv2.findNonZero(bin_)
    if nz is None:
        return bgr                      # 전부 흰색이면 그대로
    x, y, w, h = cv2.boundingRect(nz)
    return bgr[y:y + h, x:x + w]


# ─────────────────────────────────────────────
# 6) 좌-하단 기준으로 붙이기
# ─────────────────────────────────────────────
def compose_bottom(spines, gap=4, bg=(255, 255, 255)):
    if not spines:
        raise ValueError("compose_bottom: 빈 리스트")
    H = max(s.shape[0] for s in spines)
    W = sum(s.shape[1] for s in spines) + gap * (len(spines) - 1)
    canvas = np.full((H, W, 3), bg, np.uint8)
    x = 0
    for sp in spines:
        h, w = sp.shape[:2]
        canvas[H - h:H, x:x + w] = sp
        x += w + gap
    return canvas
