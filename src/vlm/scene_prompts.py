"""Scene/operation descriptions and prompt construction for the VLM stages.

Per the proposal, scene descriptions are generated from the operation names
plus a one-time manual confirmation: defaults below cover the known
operations of the reference course; `configs/scene_descriptions.json`
(op name -> description) overrides/extends them. Bump PROMPT_VERSION whenever
prompt wording changes — it is part of the cache key.
"""
from __future__ import annotations

import json
from pathlib import Path

PROMPT_VERSION = "v1"

DEFAULT_CONFIG_PATH = Path("configs/scene_descriptions.json")

# observable behavior per operation name (Vietnamese, matched case-insensitively)
DEFAULT_OP_DESCRIPTIONS: dict[str, str] = {
    "Lấy nẹp đỡ đặt vào chân vịt": "hai tay cầm dải nẹp đưa vào dưới chân vịt máy may, căn vị trí bắt đầu may",
    "Lại mũi bằng nút nhấn": "tay phải nhấn nút lại mũi trên thân máy, vải gần như đứng yên, mũi kim chạy tới-lui một đoạn ngắn",
    "Diễu đầu nẹp đỡ": "may một đoạn ngắn ở đầu nẹp, hai tay giữ vải sát chân vịt",
    "Điều chỉnh mép": "các ngón tay chỉnh lại mép vải/nẹp cho thẳng ngay trước hoặc trong khi may",
    "Lại mũi bằng cần gạt": "tay phải gạt cần lại mũi (cần gạt kim loại trên đầu máy) xuống rồi thả ra",
    "Xoay quay kim (xoay góc)": "dừng may, hai tay xoay chi tiết quanh kim đang cắm để đổi hướng đường may",
    "Đẩy cữ vào nẹp đỡ": "tay đẩy cữ (thanh dẫn hướng) áp sát vào mép nẹp trước khi may cạnh dài",
    "Diễu cạnh dài": "may liên tục dọc cạnh dài của nẹp, vải chạy đều qua chân vịt, hai tay dẫn vải",
    "Đẩy cữ ra khỏi nẹp đỡ": "tay gạt cữ (thanh dẫn hướng) tách ra khỏi mép nẹp",
    "Diễu góc tròn": "may chậm theo đường cong/góc tròn, hai tay xoay vải liên tục theo đường may",
    "Điều chỉnh mép + Diễu đầu nẹp đỡ còn lại": "chỉnh mép rồi may nốt đoạn ngắn ở đầu nẹp còn lại",
    "Diễu đầu nẹp đỡ còn lại": "may đoạn ngắn ở đầu nẹp còn lại (cuối đường may)",
    "Cắt chỉ": "cắt chỉ khi kết thúc đường may (nhấn nút cắt chỉ tự động hoặc dùng kéo)",
    "Đưa ra sau may": "kéo chi tiết ra khỏi khu vực kim sau khi may xong",
    "Kiểm tra": "hai tay nhấc chi tiết lên soi/kiểm tra đường may, máy không chạy",
    "Đưa chi tiết ra ngoài": "đưa chi tiết đã may ra ngoài khu vực làm việc",
    "UNKNOWN": "thao tác trung gian không xác định (di chuyển tay, chuẩn bị)",
}


def load_op_descriptions(config_path: str | Path | None = None) -> dict[str, str]:
    """Defaults merged with the optional manual-confirmation config file."""
    descriptions = dict(DEFAULT_OP_DESCRIPTIONS)
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.exists():
        overrides = json.loads(path.read_text(encoding="utf-8"))
        descriptions.update({k: v for k, v in overrides.items() if isinstance(v, str)})
    return descriptions


def describe_op(op_name: str, descriptions: dict[str, str]) -> str:
    for name, desc in descriptions.items():
        if name.lower() == op_name.lower():
            return desc
    return op_name  # fallback: the operation name itself is the description


def scene_catalog_text(scenes, descriptions: dict[str, str]) -> str:
    """Numbered scene list (name + observable-behavior description) that the
    classification prompt embeds. Scene order == expert execution order."""
    lines = []
    for sc in scenes:
        op_descs = "; ".join(describe_op(op, descriptions) for op in sc.operations)
        dur = sc.end - sc.start
        lines.append(f"{sc.scene_index}. \"{sc.label}\" (~{dur:.1f}s): {op_descs}")
    return "\n".join(lines)


def build_classify_messages(scenes, descriptions: dict[str, str],
                            segment_id: str, frame_b64s: list[str],
                            ref_frames: list[tuple[int, str]] | None = None) -> list[dict]:
    """Messages for tier-1 segment classification.

    `frame_b64s`: keyframes (ROI crops) of one worker candidate segment.
    `ref_frames`: optional few-shot [(scene_index, jpeg_b64), ...] reference
    crops from the expert video.
    Output contract (proposal §3.1):
      {"segment_id": ..., "scores": {"<scene_index>": 0..1, ...}, "evidence": "..."}
    """
    from src.vlm.openrouter_client import image_content, text_content

    system = (
        "Bạn là chuyên gia phân tích thao tác may công nghiệp. Bạn xem các khung hình "
        "(đã crop vào vùng thao tác quanh kim/chân vịt/hai tay) từ video một công nhân "
        "đang thực hiện công đoạn may, và xác định thao tác đó ứng với scene nào trong "
        "quy trình chuẩn của chuyên gia. Chỉ trả lời bằng một object JSON, không thêm chữ nào khác."
    )
    catalog = scene_catalog_text(scenes, descriptions)
    content: list[dict] = [text_content(
        "Danh sách scene chuẩn (theo thứ tự thực hiện của chuyên gia):\n"
        f"{catalog}\n\n"
    )]
    if ref_frames:
        content.append(text_content("Ảnh tham chiếu từ video chuyên gia (mỗi ảnh ứng với một scene):"))
        for scene_idx, b64 in ref_frames:
            content.append(text_content(f"[Tham chiếu scene {scene_idx}]"))
            content.append(image_content(b64))
    content.append(text_content(
        f"\nDưới đây là {len(frame_b64s)} khung hình liên tiếp (cách đều nhau) của MỘT đoạn video "
        f"công nhân (segment \"{segment_id}\"). Hãy chấm điểm mức độ phù hợp của đoạn này với "
        "TỪNG scene (0 = chắc chắn không phải, 1 = chắc chắn đúng). Các scene lặp lại cùng thao tác "
        "có thể có điểm bằng nhau. Trả về JSON đúng định dạng:\n"
        '{"segment_id": "' + segment_id + '", "scores": {"0": 0.0, "1": 0.0, ...}, '
        '"evidence": "mô tả ngắn những gì quan sát được"}\n'
        f'"scores" phải có đủ {len(scenes)} khóa (0 đến {len(scenes) - 1}).'))
    for b64 in frame_b64s:
        content.append(image_content(b64))
    return [{"role": "system", "content": system},
            {"role": "user", "content": content}]


def build_aux_check_messages(op_name: str, op_description: str,
                             frame_b64s: list[str], scene_label: str) -> list[dict]:
    """Messages for a tier-2 directed yes/no question: did the worker perform
    the (sub-second) auxiliary operation in these frames?
    Output contract: {"answer": "yes"|"no"|"uncertain", "confidence": 0..1,
                      "evidence": "..."}"""
    from src.vlm.openrouter_client import image_content, text_content

    system = (
        "Bạn là chuyên gia phân tích thao tác may công nghiệp. Bạn xem một cụm khung hình "
        "liên tiếp (full fps, đã crop vào vùng thao tác) và trả lời một câu hỏi CÓ/KHÔNG "
        "về một thao tác phụ rất ngắn. Chỉ trả lời bằng một object JSON."
    )
    content: list[dict] = [text_content(
        f"Đoạn video này được căn với scene \"{scene_label}\". "
        f"Câu hỏi: trong các khung hình dưới đây, công nhân CÓ thực hiện thao tác "
        f"\"{op_name}\" ({op_description}) không?\n"
        "Trả về JSON: {\"answer\": \"yes\"|\"no\"|\"uncertain\", "
        "\"confidence\": 0..1, \"evidence\": \"mô tả ngắn\"}\n"
        "Chỉ trả lời \"yes\"/\"no\" khi thấy bằng chứng rõ; nếu không chắc, trả \"uncertain\".")]
    for b64 in frame_b64s:
        content.append(image_content(b64))
    return [{"role": "system", "content": system},
            {"role": "user", "content": content}]
