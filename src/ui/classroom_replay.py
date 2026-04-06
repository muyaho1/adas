import html
import json
import os
from pathlib import Path


TEMPLATE_PATH = Path(__file__).with_name("classroom_replay_template.html")


def export_replay_html(log_data: dict, output_path: str, title: str = "Classroom ADHD Simulation") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    safe_title = html.escape(title)
    safe_payload = json.dumps(log_data, ensure_ascii=False).replace("</script>", "<\\/script>")
    document = template.replace("$title", safe_title).replace("$payload", safe_payload)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(document)
