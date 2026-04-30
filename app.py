import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import List

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()
KIE_BASE = "https://api.kie.ai"
KIE_FILE_UPLOAD_URL = os.getenv("KIE_FILE_UPLOAD_URL", f"{KIE_BASE}/api/v1/files/upload")

HEADERS_JSON = {
    "Authorization": f"Bearer {KIE_API_KEY}",
    "Content-Type": "application/json",
}
HEADERS_AUTH = {"Authorization": f"Bearer {KIE_API_KEY}"}

app = FastAPI()
templates = Jinja2Templates(directory="templates")
BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)


def split_script(script: str, max_words: int = 18) -> List[str]:
    script = re.sub(r"\s+", " ", script.strip())
    if not script:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", script)
    chunks: List[str] = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence.strip())
            continue

        parts = re.split(r"(?<=[,;:])\s+", sentence)
        current: List[str] = []
        count = 0
        for part in parts:
            part_words = part.split()
            if count + len(part_words) <= max_words:
                current.append(part)
                count += len(part_words)
            else:
                if current:
                    chunks.append(" ".join(current).strip())

                if len(part_words) > max_words:
                    for i in range(0, len(part_words), max_words):
                        chunks.append(" ".join(part_words[i : i + max_words]).strip())
                    current = []
                    count = 0
                else:
                    current = [part]
                    count = len(part_words)

        if current:
            chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def build_prompt(dialogue_chunk: str) -> str:
    return (
        "Realistic UGC talking-head video, natural indoor lighting, deep focus, no filters. "
        "DO NOT ZOOM IN OR OUT at any point. Keep exact same frame from start to finish. "
        "Eyes naturally on camera with normal blinking. "
        f'[GENERATE NATIVE AUDIO AND LIP-SYNC TO EXACT DIALOGUE]: "{dialogue_chunk}"'
    )


def upload_image_to_kie(local_path: Path) -> str:
    with open(local_path, "rb") as file_handle:
        files = {"file": (local_path.name, file_handle, "image/png")}
        response = requests.post(
            KIE_FILE_UPLOAD_URL,
            headers={"Authorization": f"Bearer {KIE_API_KEY}"},
            files=files,
            timeout=120,
        )
    response.raise_for_status()
    data = response.json()

    url = (
        data.get("data", {}).get("url")
        or data.get("data", {}).get("fileUrl")
        or data.get("data", {}).get("file_url")
        or data.get("url")
    )
    if not url:
        raise RuntimeError(f"Could not find uploaded file URL in response: {data}")
    return url


def create_kling_task(prompt: str, image_url: str, duration: str) -> str:
    payload = {
        "model": "kling-3.0/video",
        "input": {
            "prompt": prompt,
            "image_urls": [image_url],
            "sound": True,
            "duration": str(duration),
            "aspect_ratio": "9:16",
            "mode": "pro",
            "multi_shots": False,
        },
    }
    response = requests.post(
        f"{KIE_BASE}/api/v1/jobs/createTask", headers=HEADERS_JSON, json=payload, timeout=120
    )
    response.raise_for_status()
    data = response.json()
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"No taskId in response: {data}")
    return task_id


def poll_task(task_id: str, max_wait_sec: int = 900):
    start = time.time()
    while time.time() - start < max_wait_sec:
        response = requests.get(
            f"{KIE_BASE}/api/v1/jobs/recordInfo",
            headers=HEADERS_AUTH,
            params={"taskId": task_id},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        status = data.get("data", {}).get("status") or data.get("data", {}).get("taskStatus")
        status_str = str(status).upper() if status is not None else ""

        if "SUCCESS" in status_str or status_str in {"4", "SUCCEEDED"}:
            video_url = (
                data.get("data", {}).get("videoUrl")
                or data.get("data", {}).get("url")
                or data.get("data", {}).get("output", {}).get("url")
            )
            if not video_url:
                maybe_urls = data.get("data", {}).get("urls") or []
                if maybe_urls:
                    video_url = maybe_urls[0]
            if not video_url:
                raise RuntimeError(f"Task succeeded but no video URL found: {data}")
            return video_url

        if "FAIL" in status_str or status_str in {"5", "FAILED"}:
            raise RuntimeError(f"Kling task failed: {data}")

        time.sleep(5)

    raise TimeoutError(f"Task timeout after {max_wait_sec}s for taskId={task_id}")


def download_file(url: str, out_path: Path):
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def stitch_clips_ffmpeg(clip_paths: List[Path], out_path: Path):
    list_file = out_path.parent / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as file_handle:
        for clip_path in clip_paths:
            file_handle.write(f"file '{clip_path.as_posix()}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out_path),
        ],
        check=True,
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(script: str = Form(...), image_file: UploadFile = File(...)):
    if not KIE_API_KEY:
        return JSONResponse({"ok": False, "error": "Missing KIE_API_KEY env var"}, status_code=500)

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    clips_dir = job_dir / "clips"
    job_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(exist_ok=True)

    image_path = job_dir / image_file.filename
    image_path.write_bytes(await image_file.read())

    try:
        image_url = upload_image_to_kie(image_path)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Image upload failed: {exc}"}, status_code=500)

    chunks = split_script(script, max_words=18)
    if not chunks:
        return JSONResponse({"ok": False, "error": "Empty script"}, status_code=400)

    results = []
    successful_paths: List[Path] = []

    for index, chunk in enumerate(chunks, start=1):
        prompt = build_prompt(chunk)
        words = len(chunk.split())
        duration = str(max(3, min(7, round(words / 2.6))))

        try:
            task_id = create_kling_task(prompt, image_url, duration)
            video_url = poll_task(task_id)
            clip_path = clips_dir / f"clip_{index:03d}.mp4"
            download_file(video_url, clip_path)
            successful_paths.append(clip_path)

            results.append(
                {
                    "index": index,
                    "duration": duration,
                    "chunk": chunk,
                    "task_id": task_id,
                    "video_url": video_url,
                    "clip_file": clip_path.name,
                }
            )
        except Exception as exc:
            results.append({"index": index, "duration": duration, "chunk": chunk, "error": str(exc)})

    final_video = None
    if successful_paths:
        final_path = job_dir / "final.mp4"
        try:
            stitch_clips_ffmpeg(successful_paths, final_path)
            final_video = f"/jobs/{job_id}/final.mp4"
        except Exception as exc:
            results.append({"stitch_error": str(exc)})

    meta = {
        "ok": True,
        "job_id": job_id,
        "image_url": image_url,
        "chunks": chunks,
        "results": results,
        "final_video": final_video,
    }
    (job_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


@app.get("/jobs/{job_id}/{filename}")
def get_job_file(job_id: str, filename: str):
    path = JOBS_DIR / job_id / filename
    if path.exists():
        return FileResponse(path)
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.get("/jobs/{job_id}/clips/{clip_name}")
def get_clip(job_id: str, clip_name: str):
    path = JOBS_DIR / job_id / "clips" / clip_name
    if path.exists():
        return FileResponse(path)
    return JSONResponse({"error": "Not found"}, status_code=404)
