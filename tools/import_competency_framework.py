"""Import khung năng lực (competency framework) vào bảng competency_framework.

Framework là dữ liệu MẬT: KHÔNG có endpoint API nào thêm/sửa/xoá nó — muốn cập nhật
phải chạy lại script này với file JSON mới. Service chỉ ĐỌC (sinh đề / chấm / trả
cấu trúc cho BE).

File nguồn mặc định: FPT_Math_Competency_Framework_MAE101_MAD101_MAS291_v1.json
(ở repo root). Tách thành 3 loại bản ghi:
  - kind='shared_policy' : shared_policy (assessment_modes, ai_assistance_levels,
                           confidence_policy, method_scope_levels, ...)
  - kind='gmc'           : general_mathematical_competency_framework (10 dimensions)
  - kind='subject'       : mỗi môn 1 bản ghi (MAE101/MAD101/MAS291) — curriculum_units,
                           clos→skills, general_math_competency_profile, essay_rubric,
                           question_generation_profile

Chạy:
  python -m tools.import_competency_framework                # import file mặc định
  python -m tools.import_competency_framework --file path.json
  python -m tools.import_competency_framework --dry-run      # chỉ đếm, không ghi DB
"""
import argparse
import json
import os
import sys

# cho phép chạy như script trực tiếp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AI_service import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(ROOT, "FPT_Math_Competency_Framework_MAE101_MAD101_MAS291_v1.json")


def load(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def summarize(data: dict) -> None:
    subjects = data.get("subjects") or {}
    gmc = (data.get("general_mathematical_competency_framework") or {}).get("dimensions") or []
    print(f"framework_id : {data.get('framework_id')}")
    print(f"version      : {data.get('schema_version')}")
    print(f"GMC dims     : {len(gmc)}")
    for code, s in subjects.items():
        clos = s.get("clos") or []
        skills = sum(len(c.get("skills") or []) for c in clos)
        units = len(s.get("curriculum_units") or [])
        print(f"  {code}: {len(clos)} CLO, {units} unit, {skills} skill")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = load(args.file)
    framework_id = data.get("framework_id") or "UNKNOWN_FRAMEWORK"
    version = str(data.get("schema_version") or "0")
    summarize(data)
    if args.dry_run:
        print("(dry-run — không ghi DB)")
        return

    db.init_db()

    shared_policy = data.get("shared_policy") or {}
    db.upsert_framework_part(framework_id=framework_id, version=version,
                             kind="shared_policy", payload=shared_policy)

    gmc = data.get("general_mathematical_competency_framework") or {}
    db.upsert_framework_part(framework_id=framework_id, version=version,
                             kind="gmc", payload=gmc)

    subjects = data.get("subjects") or {}
    for code, payload in subjects.items():
        db.upsert_framework_part(framework_id=framework_id, version=version,
                                 kind="subject", subject_code=code, payload=payload)

    print(f"Đã import: shared_policy + gmc + {len(subjects)} môn "
          f"({', '.join(subjects.keys())}) — framework {framework_id} v{version}")


if __name__ == "__main__":
    main()
