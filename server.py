import uuid, cv2, base64, os, time, re, requests
import json
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from rapidfuzz import process, fuzz
from konlpy.tag import Okt
import face_recognition
#from fuzzywuzzy import
import oracledb
from routers.detection import detect_polygons
from models.spine_processor import (
    load_polygon_labels_from_text,
    rectify_spine_roi, trim_whitespace,
    compose_bottom
)

# ── 경로 설정 ─────────────────────────────────────
ROOT = Path(__file__).parent
STATIC = ROOT / "static"
SP_DIR = STATIC / "spines"
SP_DIR.mkdir(parents=True, exist_ok=True)
CMP_DIR = STATIC / "compose"
CMP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = STATIC / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
CORS(app)

DB_USER = "c##webproj"
DB_PASSWORD = "webproj"
DB_DSN = "localhost:1521/XE"

try:
    pool = oracledb.create_pool(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN, min=1, max=5, increment=1)
    print("DB 연결 풀 생성 성공")
except Exception as e:
    print(f"DB 연결 풀 생성 실패: {e}")
    pool = None

def get_db_connection():
    if pool:
        return pool.acquire()
    raise Exception("DB 연결 풀이 초기화되지 않았습니다.")


# ── 이미지 디코딩 유틸 ─────────────────────────────
def _read_cv2(fs) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_COLOR)

# ── PaddleOCR 및 유틸 초기화 ──────────────────────
#paddle_ocr = PaddleOCR(use_angle_cls=True, lang='korean')
okt = Okt()

NAVER_CLIENT_ID = 'KcOstkmsDUomWuRjbKH7'
NAVER_CLIENT_SECRET = 'iSC_Jj7JKK'

correction_dict = {
        # 흔히 OCR을 통해 잘못 읽힐 수 있는 키워드: 올바른 도서명
    "머신러님": "머신러닝",
    "덥러님": "딥러닝",
    "답러님": "딥러닝",
    "신러닝": "머신러닝",

    # 왼쪽 책들 (OCR에서 잘못 나올 가능성이 있는 키를, 정확한 제목으로 매핑)
    "opencv": "OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝",
    "머신러님": "머신러닝 교과서",
    "aws": "AWS 교과서",
    "이론과실습": "머신러닝 이론과 실습",
    "api챗봇": "챗GPT API를 활용한 챗봇 만들기",
    "모두의딥러닝": "모두의 딥러닝",
    "파이썬분석": "파이썬 빅데이터 분석 기초와 실습",
    "텍스트분석": "모두의 한국어 텍스트 분석",
    "딥러닝챗봇": "처음 배우는 딥러닝 챗봇",

    # 오른쪽 책들
    "네트워크기초": "모두의 네트워크 기초",
    "운영제": "운영체제",
    "하이테크": "하이테크 교실 수업",
    "프롬프트": "프롬프트 엔지니어",
    "자동화": "챗GPT와 업무자동화",
    "파이썬40가지": "챗GPT를 활용한 40가지 파이썬 프로그램 만들기",
    "챗gpt101": "챗GPT 101",
    "정보통신개론": "4차 산업혁명 시대의 정보통신개론",
    "자바프로그래밍": "JAVA Programming",
    "자바사": "JAVA 자바",
    "자바": "JAVA 자바"
}

def is_korean_text(text):
    return len(re.findall(r'[가-힣]', text)) >= 2

def extract_nouns(text):
    return [w for w in okt.nouns(text) if len(w) >= 2]

def correct_by_dict(ocr_text: str, threshold: int = 60):
    """
    ocr_text (예: '뉴금밀남동교이어')와 correction_dict 키들 중
    rapidfuzz로 비교 → score가 threshold 이상이면 해당 키의 correction_dict 값 반환
    """
    if not ocr_text or not correction_dict:
        return None, 0

    # 1) OCR 결과를 소문자, 공백 없애기 등 간단하게 정규화
    norm = ocr_text.lower().replace(" ", "").replace("_", "")

    # 2) correction_dict 키들만 리스트로 꺼내서 rapidfuzz 비교
    keys = list(correction_dict.keys())
    match_key, score, _ = process.extractOne(
        norm, keys, scorer=fuzz.WRatio
    )
    if score >= threshold:
        # 최종적으로 “치환된 텍스트”는 correction_dict[match_key]
        return correction_dict[match_key], score
    else:
        return None, score

def run_naver_ocr(image_np: np.ndarray) -> list:
    import uuid
    api_url = 'https://zz03r4vqrs.apigw.ntruss.com/custom/v1/42879/a5e411b62f58cdde294f8e188bbf4cea7b1bf0bb9d66748ec4d66351f48e0faa/general'
    secret_key = 'a2tvd2RDenRCakdFb2JWekNFRFNJUG9QTHNoVHppZHU='

    _, img_encoded = cv2.imencode(".jpg", image_np)
    image_bytes = img_encoded.tobytes()

    request_json = {
        "images": [{"format": "jpg", "name": "book"}],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(time.time() * 1000)
    }

    payload = {"message": json.dumps(request_json).encode("utf-8")}
    files = [("file", ("book.jpg", image_bytes, "image/jpeg"))]
    headers = {"X-OCR-SECRET": secret_key}

    response = requests.post(api_url, headers=headers, data=payload, files=files)
    result = response.json()

    texts = []
    for field in result.get("images", [])[0].get("fields", []):
        txt = field.get("inferText", "").strip()
        if txt:
            texts.append(txt)
    return texts

def search_books_naver(query):
    time.sleep(1)  # 429 방지
    url = 'https://openapi.naver.com/v1/search/book.json'
    headers = {
        'X-Naver-Client-Id': NAVER_CLIENT_ID,
        'X-Naver-Client-Secret': NAVER_CLIENT_SECRET
    }
    params = {'query': query, 'display': 10}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        return [item['title'].replace('<b>', '').replace('</b>', '')
                for item in resp.json().get('items', [])]
    return []

def correct_ocr_result(ocr_text, candidate_titles):
    #OCR 결과 텍스트와 후보 도서 제목 리스트를 비교하여
    #가장 유사한 제목 반환 (50% 이상인 경우만 인정)
    if not candidate_titles:
        return None
    match_result = process.extractOne(ocr_text, candidate_titles)
    if match_result is None:
        return None
    best_match, score, _ = match_result #튜플에서 3가지 값 받기
    print(f"[MATCH] '{ocr_text}' → '{best_match}' (유사도 {score}%)")
    return best_match if score >= 50 else None

def find_best_book_title(ocr_text: str):
    corrected_text, _ = correct_by_dict(ocr_text)
    primary_query = corrected_text or ocr_text
    nouns = extract_nouns(primary_query)
    noun_query = " ".join(nouns)
    query_candidates = [q for q in [primary_query, noun_query] if q.strip()]
    for query in query_candidates:
        candidate_titles = search_books_naver(query)
        if not candidate_titles: continue
        match_result = process.extractOne(primary_query, candidate_titles, scorer=fuzz.WRatio)
        if match_result:
            best_match, score, _ = match_result
            if score >= 85:
                return {"matched_title": best_match, "similarity_score": round(score), "used_query": query, "recognition_status": "success"}
    return None

@app.route("/rearrange", methods=["POST"])
def rearrange():
    if "image" not in request.files:
        return jsonify({"status": "error", "msg": "image 필수"}), 400

    img = _read_cv2(request.files["image"])
    h_img, w_img = img.shape[:2]
    order = request.form.get("order", "asc").lower()
    gap = int(request.form.get("gap", 4))
    if order not in ("asc", "desc"):
        order = "asc"

    if "labels" in request.files:
        txt = request.files["labels"].read().decode("utf-8")
        polys = load_polygon_labels_from_text(txt, img.shape)
    else:
        polys = detect_polygons(img)

    spines, spine_urls = [], []
    for poly in polys:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        box_w, box_h = max(xs) - min(xs), max(ys) - min(ys)
        if box_w > 0.5 * w_img and box_h > 0.5 * h_img:
            continue

        roi = trim_whitespace(rectify_spine_roi(img, poly))
        if roi.size == 0:
            continue
        if roi.shape[0] < 20 or roi.shape[1] < 10:
            continue

        fname = f"{uuid.uuid4().hex}.png"
        cv2.imwrite(str(SP_DIR / fname), roi)
        spine_urls.append(request.url_root.rstrip("/") + f"/static/spines/{fname}")
        spines.append(roi)

    if not spines:
        return jsonify({"status": "error", "msg": "검출 실패"}), 200

    spines_sorted = sorted(spines, key=lambda s: s.shape[0], reverse=(order == "desc"))
    composite = compose_bottom(spines_sorted, gap=gap)
    cmp_name = f"{uuid.uuid4().hex}.png"
    cv2.imwrite(str(CMP_DIR / cmp_name), composite)
    composite_url = request.url_root.rstrip("/") + f"/static/compose/{cmp_name}"

    return jsonify({
        "status": "success",
        "order": order,
        "gap": gap,
        "spines": spine_urls,
        "compose": composite_url
    }), 200

@app.route("/ocr-search", methods=["POST"])
def ocr_search():
    images      = request.files.getlist("image")
    label_files = request.files.getlist("label")
    print(f"[DEBUG] label files count: {len(label_files)}")

    results = []
    for idx, fs in enumerate(images):
        img = cv2.imdecode(
            np.frombuffer(fs.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # (A) CSV 라벨 시도
        if idx < len(label_files):
            txt = label_files[idx].read().decode("utf-8")
            from models.spine_processor import load_polygon_labels_from_text
            polys = load_polygon_labels_from_text(txt, img.shape)
        else:
            # CSV 없으면 YOLO fallback
            from routers.detection import detect_polygons
            polys = detect_polygons(img)
            print(f"[DEBUG] CSV 파싱 실패, YOLO fallback → detected {len(polys)} spine boxes")

        if not polys:
            print(f"[WARN] 이미지 {idx}에 대해 CSV도 YOLO도 spine 검출 실패")
            continue

        for poly in polys:
            from models.spine_processor import rectify_spine_roi, trim_whitespace
            roi = rectify_spine_roi(img, poly)
            if roi is None or roi.size == 0:
                # ─── 기존: ROI 가 유효하지 않으면 건너뛰고 OCR 결과에도 포함하지 않음 ───
                # continue

                # ─── 변경: ROI 자체가 유효하지 않은 경우에도 “빈 OCR” 형태로 결과에 추가 ───
                results.append({
                    "ocr_texts":    [],              # OCR 라인별 원문: 빈 리스트
                    "ocr_string":   "",              # 합쳐진 문자열: 빈 문자열
                    "corrected":    "",              # 사전 교정: 없음
                    "used_query":   "",              # 사용 쿼리: 없음
                    "candidates":   [],              # 검색 후보: 없음
                    "matched_title":"",              # 추천 도서: 없음
                    "similarity_score": 0            # 유사도 점수: 0
                })
                continue

            roi = trim_whitespace(roi)
            if roi.shape[0] < 20 or roi.shape[1] < 10:
                # ─── 기존: 크기가 너무 작으면 건너뛰고 OCR 결과에도 포함하지 않음 ───
                # continue

                # ─── 변경: 크기가 너무 작아도 “빈 OCR” 형태로 결과에 추가 ───
                results.append({
                    "ocr_texts":    [],
                    "ocr_string":   "",
                    "corrected":    "",
                    "used_query":   "",
                    "candidates":   [],
                    "matched_title":"",
                    "similarity_score": 0
                })
                continue

            # ─── PaddleOCR 수행 ───
            texts = run_naver_ocr(roi)
            print(f"[OCR DEBUG] spine ROI result: {texts }")

            # ─── 기존: OCR 결과가 없거나 형식이 올바르지 않으면 건너뛰기 ───
            # if not ocr_result or not isinstance(ocr_result[0], list):
            #     continue

            # ─── 추출된 텍스트가 하나도 없으면 빈 결과로 추가 ───
            if not texts:
                results.append({
                    "ocr_texts":    [],
                    "ocr_string":   "",
                    "corrected":    "",
                    "used_query":   "",
                    "candidates":   [],
                    "matched_title":"",
                    "similarity_score": 0
                })
                continue

            merged_text = " ".join(texts)  # ex: “뉴금밀남동교이어”

            results.append({
                "ocr_texts":    texts,          # OCR 라인별 원본 문자열 리스트
                "ocr_string":   merged_text,    # 합쳐진 OCR 문자열
                "corrected":    "",             # 사전 교정 없음
                "used_query":   "",             # 검색 안 함
                "candidates":   [],             # 검색 후보 없음
                "matched_title":"",             # 추천 도서 없음
                "similarity_score": 0           # 유사도 점수 없음
            })

    return jsonify({
        "status": "success",
        "results": results
    }), 200

@app.route("/shelf-upload", methods=["POST"])
def shelf_upload():
    conn, cursor = None, None
    try:
        user_id = request.form.get("user_id")
        shelf_name = request.form.get("shelf_name")
        shelf_rows = int(request.form.get("shelf_rows", 0))
        shelf_cols = int(request.form.get("shelf_cols", 0))

        if not user_id: return jsonify({"status": "error", "msg": "user_id가 필요합니다."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        
        shelf_id_var = cursor.var(oracledb.NUMBER)
        cursor.execute(
            "INSERT INTO TBL_BOOKSHELVES (user_id, shelf_name, shelf_rows, shelf_cols) VALUES (:1, :2, :3, :4) RETURNING shelf_id INTO :5",
            [user_id, shelf_name, shelf_rows, shelf_cols, shelf_id_var]
        )
        shelf_id = shelf_id_var.getvalue()[0]
        
        cell_files_list = request.files.getlist("cells")
        cells_map = {}
        for fs in cell_files_list:
            fname = fs.filename or ""; coord, ftype = fname.rsplit("_", 1); r, c = map(int, coord.split("_")); cells_map.setdefault((r, c), {})[ftype] = fs

        for (r, c), file_dict in cells_map.items():
            img_fs = file_dict.get("image")
            if not img_fs: continue

            # --- [수정된 로직] ---
            # 1. 먼저 파일을 디스크에 완전히 저장합니다.
            filename = f"{uuid.uuid4().hex}.jpg"
            save_path = UPLOAD_DIR / filename
            img_fs.save(str(save_path))
            my_ip_address = "172.31.57.36" 
            image_url = f"http://{my_ip_address}:8000/static/uploads/{filename}"
            
            # 2. 저장된 파일 경로를 TBL_CELLS에 INSERT 합니다.
            cursor.execute(
                "INSERT INTO TBL_CELLS (shelf_id, cell_row, cell_col, img_url) VALUES (:shelf_id, :cell_row, :cell_col, :img_url)",
                {"shelf_id": shelf_id, "cell_row": r, "cell_col": c, "img_url": image_url}
            )
            
            # 3. OCR/탐지 작업은 방금 저장된 파일을 다시 읽어서 수행합니다. (스트림 재사용 방지)
            img = cv2.imread(str(save_path))
            # --- [수정된 로직 끝] ---

            polys = detect_polygons(img)
            
            book_order_in_cell = 0
            for poly in polys:
                book_order_in_cell += 1
                roi = rectify_spine_roi(img, poly)
                if roi is None or roi.size == 0: continue
                roi = trim_whitespace(roi)
                if roi.shape[0] < 20 or roi.shape[1] < 10: continue

                ocr_texts = run_naver_ocr(roi)
                merged_text = " ".join(ocr_texts)

                cursor.execute(
                    "INSERT INTO TBL_BOOKS (shelf_id, cell_row, cell_col, book_order, ocr_text, final_title) VALUES (:1, :2, :3, :4, :5, :6)",
                    [shelf_id, r, c, book_order_in_cell, merged_text, merged_text]
                )

        conn.commit()
        return jsonify({"status": "success", "msg": "책장 생성 및 모든 데이터 저장 완료"}), 200

    except Exception as e:
        if conn: conn.rollback()
        print(f"[/shelf-upload] 오류 발생: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn and pool: pool.release(conn)

@app.route("/bookshelves/<user_id>", methods=["GET"])
def get_user_bookshelves(user_id):
    shelves = []
    conn = None # conn 변수 초기화
    cursor = None # cursor 변수 초기화
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # ▼▼▼ [수정] SELECT 쿼리의 컬럼명을 shelf_rows, shelf_cols로 변경 ▼▼▼
        cursor.execute("""
            SELECT shelf_id, shelf_name, shelf_rows, shelf_cols, created_at 
            FROM TBL_BOOKSHELVES 
            WHERE user_id = :1 
            ORDER BY created_at DESC
        """, [user_id])
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        for row in cursor:
            shelves.append({
                "shelf_id": row[0],
                "shelf_name": row[1],
                "rows": row[2],       # DB의 shelf_rows 값을 'rows' 키로 매핑
                "cols": row[3],       # DB의 shelf_cols 값을 'cols' 키로 매핑
                "created_at": row[4]
            })
        return jsonify({"status": "success", "bookshelves": shelves}), 200
    except Exception as e:
        print(f"[/bookshelves] 오류 발생: {e}") # 로그 추가
        return jsonify({"status": "error", "msg": str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn and pool: pool.release(conn)

@app.route("/correct-title", methods=["POST"])
def correct_title():
    """
    클라이언트로부터 받은 OCR 텍스트 한 줄을 보정/검색하여
    가장 유력한 책 제목 후보를 JSON으로 반환합니다.
    Request: {"ocr_text": "인식된 텍스트"}
    """
    # 1. 클라이언트로부터 JSON 데이터 수신
    data = request.get_json()
    if not data or "ocr_text" not in data:
        return jsonify({"status": "error", "msg": "ocr_text가 필요합니다."}), 400

    ocr_text = data["ocr_text"]
    if not ocr_text.strip():
        return jsonify({
            "ocr_string": ocr_text,
            "matched_title": "",
            "recognition_status": "failed",
            "msg": "입력 텍스트가 비어있습니다."
        }), 200

    # 2. 기존에 만들어둔 책 제목 찾기 파이프라인 실행
    correction_result = find_best_book_title(ocr_text)

    # 3. 결과에 따라 응답 구성
    if correction_result:
        # 성공 시: 파이프라인 결과와 OCR 원문을 합쳐서 반환
        final_result = {"ocr_string": ocr_text, **correction_result}
        return jsonify(final_result), 200
    else:
        # 실패 시
        return jsonify({
            "ocr_string": ocr_text,
            "matched_title": "",
            "similarity_score": 0,
            "used_query": "",
            "recognition_status": "failed"
        }), 200

@app.route("/shelf-details/<int:shelf_id>", methods=["GET"])
def get_shelf_details(shelf_id):
    conn, cursor = None, None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cells_data = {}
        cursor.execute("SELECT cell_row, cell_col, img_url FROM TBL_CELLS WHERE shelf_id = :1", [shelf_id])
        for row in cursor:
            cells_data[(row[0], row[1])] = {"cell_row": row[0], "cell_col": row[1], "image_url": row[2], "books": []}

        cursor.execute("SELECT cell_row, cell_col, book_id, final_title, ocr_text FROM TBL_BOOKS WHERE shelf_id = :1", [shelf_id])
        for row in cursor:
            if (row[0], row[1]) in cells_data:
                cells_data[(row[0], row[1])]["books"].append({"book_id": row[2], "final_title": row[3], "ocr_text": row[4]})
        
        # ▼▼▼▼▼ 디버깅을 위한 print문 추가 ▼▼▼▼▼
        final_response_data = {"status": "success", "cells": list(cells_data.values())}
        print("서버 응답 직전 데이터:", json.dumps(final_response_data, indent=2, ensure_ascii=False))
        # ▲▲▲▲▲ 디버깅을 위한 print문 추가 ▲▲▲▲▲

        return jsonify(final_response_data)

    except Exception as e:
        print(f"[/shelf-details/{shelf_id}] 오류 발생: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn and pool: pool.release(conn)

@app.route("/books/update", methods=["POST"])
def update_book_titles():
    conn = None
    try:
        data = request.get_json()
        updates = data.get("updates") 

        if not updates:
            return jsonify({"status": "error", "msg": "업데이트할 내용이 없습니다."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # ▼▼▼▼▼ [수정된 부분] KeyError 방지를 위해 .get() 사용 ▼▼▼▼▼
        # 튜플 리스트 준비: [(final_title1, book_id1), (final_title2, book_id2), ...]
        update_data = [
            (item.get('final_title', ''), item.get('book_id'))
            for item in updates
        ]
        # ▲▲▲▲▲ [수정된 부분] ▲▲▲▲▲
        
        cursor.executemany("UPDATE TBL_BOOKS SET final_title = :1 WHERE book_id = :2", update_data)
        
        conn.commit()
        
        print(f"{len(updates)}개의 책 제목이 DB에서 업데이트되었습니다.")
        return jsonify({"status": "success", "updated_count": len(updates)}), 200

    except Exception as e:
        if conn: conn.rollback()
        print(f"[/books/update] 오류 발생: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500
    finally:
        if 'cursor' in locals() and cursor: cursor.close()
        if 'conn' in locals() and conn and pool: pool.release(conn)

@app.route("/search-books/<user_id>", methods=["GET"])
def search_all_books(user_id):
    query = request.args.get("query", "") # URL 파라미터에서 검색어 가져오기
    if not query:
        return jsonify({"status": "error", "msg": "검색어(query)가 필요합니다."}), 400

    results = []
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # TBL_BOOKS와 TBL_BOOKSHELVES를 JOIN하여 검색
        # final_title에 검색어가 포함된 책을 찾고, 그 책이 속한 책장의 이름도 함께 가져옵니다.
        cursor.execute("""
            SELECT b.final_title, s.shelf_name, b.cell_row, b.cell_col
            FROM TBL_BOOKS b
            JOIN TBL_BOOKSHELVES s ON b.shelf_id = s.shelf_id
            WHERE s.user_id = :user_id AND b.final_title LIKE '%' || :query || '%'
        """, {"user_id": user_id, "query": query})

        for row in cursor:
            results.append({
                "found_title": row[0],
                "shelf_name": row[1],
                "cell_row": row[2],
                "cell_col": row[3]
            })
            
        return jsonify({"status": "success", "search_results": results}), 200

    except Exception as e:
        print(f"[/search-books] 오류 발생: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn and pool: pool.release(conn)

@app.route("/upload-booklist", methods=["POST"])
def upload_booklist():
    if 'booklist_file' not in request.files:
        return jsonify({"status": "error", "msg": "파일이 없습니다."}), 400
    
    file = request.files['booklist_file']
    
    if file.filename == '':
        return jsonify({"status": "error", "msg": "선택된 파일이 없습니다."}), 400

    if file and file.filename.endswith('.csv'):
        # 저장할 경로 생성
        booklist_dir = STATIC / "booklist"
        booklist_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 중복을 피하기 위해 타임스탬프 추가
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        save_path = booklist_dir / filename
        
        file.save(str(save_path))
        
        file_url = request.url_root.rstrip("/") + f"/static/booklist/{filename}"
        print(f"파일 저장 성공: {save_path}")
        
        return jsonify({"status": "success", "msg": "파일 업로드 성공!", "file_url": file_url}), 200
    else:
        return jsonify({"status": "error", "msg": "CSV 파일만 업로드 가능합니다."}), 400

@app.route("/register-face", methods=["POST"])
def register_face():
    if 'image' not in request.files or 'user_id' not in request.form:
        return jsonify({"status": "error", "msg": "이미지 또는 user_id가 필요합니다."}), 400

    user_id = request.form['user_id']
    fs = request.files['image']
    
    # 전달받은 이미지 로드
    img = face_recognition.load_image_file(fs)
    
    # 이미지에서 얼굴 위치 탐색 (첫 번째 얼굴만 사용)
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        return jsonify({"status": "error", "msg": "이미지에서 얼굴을 찾을 수 없습니다."}), 400
        
    # 얼굴 특징값(128차원 벡터) 추출
    # 한 이미지에 여러 얼굴이 있어도 첫 번째 얼굴의 특징값만 사용합니다.
    face_embedding = face_recognition.face_encodings(img, known_face_locations=face_locations)[0]
    
    # 특징값(numpy 배열)을 DB에 저장하기 위해 JSON 형태의 문자열로 변환
    embedding_str = json.dumps(face_embedding.tolist())

    conn, cursor = None, None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE TBL_USERS SET face_embedding = :embedding WHERE user_id = :user_id",
            [embedding_str, user_id]
        )
        conn.commit()
        return jsonify({"status": "success", "msg": f"{user_id}님의 얼굴 정보가 등록되었습니다."})

    except Exception as e:
        if conn: conn.rollback()
        return jsonify({"status": "error", "msg": str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn and pool: pool.release(conn)

@app.route("/login-face", methods=["POST"])
def login_face():
    if 'image' not in request.files:
        return jsonify({"status": "error", "msg": "이미지가 필요합니다."}), 400

    unknown_fs = request.files['image']
    unknown_image = face_recognition.load_image_file(unknown_fs)
    
    # 1. 먼저 사진에서 얼굴이 있는지 확인
    unknown_face_locations = face_recognition.face_locations(unknown_image)
    if not unknown_face_locations:
        return jsonify({"status": "error", "msg": "이미지에서 얼굴을 찾을 수 없습니다."}), 200

    # 2. 얼굴이 있다면 특징값 추출
    unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_locations)[0]

    conn, cursor = None, None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 3. DB에서 얼굴 정보가 등록된 모든 관리자(ADMIN)의 얼굴 특징값을 가져옴
        cursor.execute("SELECT user_id, face_embedding FROM TBL_USERS WHERE role = 'ROLE_ADMIN' AND face_embedding IS NOT NULL")
        
        known_encodings = []
        known_user_ids = []
        for row in cursor:
            user_id = row[0]
            # DB의 CLOB 데이터를 읽고, JSON 문자열을 파싱하여 numpy 배열로 변환
            embedding_str = row[1].read() 
            embedding = np.array(json.loads(embedding_str))
            known_encodings.append(embedding)
            known_user_ids.append(user_id)

        if not known_encodings:
            return jsonify({"status": "error", "msg": "등록된 관리자 얼굴 정보가 없습니다."}), 200

        # 4. 현재 얼굴과 등록된 모든 얼굴을 비교
        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.4)
        
        # 5. 일치하는 얼굴이 있는지 확인
        if True in matches:
            first_match_index = matches.index(True)
            matched_user_id = known_user_ids[first_match_index]
            return jsonify({"status": "success", "user_id": matched_user_id})
        else:
            return jsonify({"status": "error", "msg": "일치하는 사용자를 찾을 수 없습니다."})

    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn and pool: pool.release(conn)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
