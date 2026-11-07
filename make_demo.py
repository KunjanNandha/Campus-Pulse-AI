"""Record and narrate a repeatable Campus Pulse AI walkthrough.

Run with: .venv\Scripts\python.exe make_demo.py
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "demo_narrated.mp4"
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / "demo_browsers"))
VOICE = os.environ.get("DEMO_VOICE", "en-IN-NeerjaNeural")
RATE = os.environ.get("DEMO_RATE", "-8%")
SIZE = {"width": 1920, "height": 1080}
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

# Edit narration here. Each action runs before its paragraph begins.
STORYBOARD = [
    {"action": "intro", "narration": "Welcome to Campus Pulse AI. Student feedback can be scattered across forms and messages, making recurring campus problems difficult to spot. This Streamlit dashboard brings comments together, classifies their sentiment, and summarizes which services may need attention. I will show the implementation from input through scoring, visual analysis, evaluation, and human review."},
    {"action": "sidebar", "narration": "The control center on the left accepts a student comment, can identify its campus service automatically, and optionally records a student rating. It also imports a CSV with a feedback column. Farther down are fine-grained sentiment thresholds, a session-only custom word lexicon, and a choice of two scoring methods. The app starts with a synthetic demonstration dataset, so these counts represent teaching examples rather than real survey evidence."},
    {"action": "tab:Overview", "narration": "The Overview tab summarizes the selected comments as positive, negative, neutral, or mixed. The mood score is a simple indicator calculated from positive minus negative comments divided by all comments. The donut chart shows the label distribution, and the charts below show feedback volume over time and frequently repeated words. Filters across the top let us focus on specific services or search for a phrase."},
    {"action": "tab:Service insights", "narration": "Service insights compares campus areas such as the canteen, Wi-Fi, transport, classrooms, and library. It shows negative share, comment volume, average score, and a follow-up queue. A service enters that queue through an explicit count and percentage rule shown beneath the alerts. This gives staff a starting point for investigation, while the original comments still need to be read before decisions are made."},
    {"action": "tab:Feedback explorer", "narration": "The Feedback explorer keeps each original comment next to its detected service, final sentiment, numeric score, aspect information, and review flags. A mixed comment may still have a positive or negative numeric score, because the mixed label comes from opposing clauses. The table also shows whether sarcasm adjustment or rating mismatch was found. Results and a short summary can be exported for further inspection."},
    {"action": "tab:Pipeline Inspector", "narration": "The Pipeline Inspector exposes how a result is produced. Its examples compare emphasis, negation, contrast, and a possible sarcasm cue. We can inspect a selected comment step by step: original text, small abbreviation expansion, word values, fired rule cues, compound score, and final label. The original wording stays available. This transparency is useful when a teacher or reviewer asks why the dashboard classified a sentence a certain way."},
    {"action": "tab:Aspects & actions", "narration": "Aspects and actions looks inside each service. A canteen comment may refer to food quality, price, hygiene, or queue speed; a Wi-Fi comment may concern speed, connectivity, or coverage. The app splits clauses and scores matched aspects separately, then displays a heat map and relevant comments. Suggestions are linked to selected service and aspect pairs. These are rule-based prompts for follow-up, not automatic decisions."},
    {"action": "tab:Comparison", "narration": "Comparison brings service results together in a leaderboard and radar chart. It also compares available student ratings with stars derived from sentiment scores, flagging large differences for review. Another section contrasts baseline VADER with the selected campus method. This matters because adding campus words and context rules can change classifications. The comparison makes those changes visible instead of hiding them inside an opaque model."},
    {"action": "tab:Terms & associations", "narration": "Terms and associations counts frequently used positive and negative dictionary words and adjacent word pairs. A service selector narrows the view to one area, such as transport or Wi-Fi. These charts help find language that appears repeatedly in feedback. They are simple counts after chart-only stop-word removal; an adjacent pair is not proof of a relationship or the reason behind a complaint."},
    {"action": "tab:Trends & warnings", "narration": "Trends and warnings displays daily average sentiment and a three-day moving average. Its early warning rule looks at two-day windows of negative comments against earlier windows for the same service. Sliders control the baseline deviation threshold, minimum total volume, and minimum negative count. The app shows the calculation so viewers can judge an alert. A warning signals a pattern worth checking, not a confirmed campus incident."},
    {"action": "tab:Limitations", "narration": "The Limitations tab demonstrates difficult wording: sarcasm, subtle negation, mixed praise and criticism, and very short comments. For example, oh great, the Wi-Fi is down again contains praise-like words alongside an explicit failure. A cautious hand-written rule can adjust that case, but it cannot understand every form of irony. Low-confidence flags send uncertain comments toward human review rather than presenting every prediction as certain."},
    {"action": "tab:Evaluation & review", "narration": "Evaluation compares baseline VADER and the two available campus scoring methods against sixty hand-labelled teaching examples. It shows agreement, a confusion matrix, per-class measures, and disagreements to study. Below that, the review queue allows a person to correct service, sentiment, or main aspect and export the corrections. The examples are small and synthetic, so their agreement numbers do not establish accuracy on real student feedback."},
    {"action": "tab:Method & project notes", "narration": "Method and project notes document the full pipeline: collect comments, preserve original wording, expand selected abbreviations for scoring, apply lexicon and context rules, score clauses, detect mixed opinions, and build charts and alerts. The default method extends VADER with campus vocabulary. Campus Context Score is a separate experimental weighted rule scorer. Neither is a trained machine-learning classifier, and both remain limited by the words and rules supplied."},
    {"action": "add_comment", "narration": "Now I will add a new example: the bus is late, but the driver is polite. The service is left on auto-detect. After submission, the dashboard adds the comment to this browser session and recalculates its summaries. Its two clauses express opposite opinions: punctuality is negative and driver behaviour is positive. This is why the final sentiment can be Mixed even if one overall numeric score leans in one direction."},
    {"action": "search_comment", "narration": "Searching for the exact new comment narrows the feedback explorer to its row. Here we can inspect the preserved original text, normalized scoring copy, numeric score, opposing clauses, and aspect result. This is an important implementation detail: a single comment may mention more than one quality of a service. It is safer to show those parts than to pretend the entire sentence has only one clear opinion."},
    {"action": "switch_algorithm", "narration": "I am switching from the default VADER plus campus lexicon to Campus Context Score. The latter uses its own weighted word calculation with nearby negation, boosters, capital letters, punctuation, and contrast rules. It is an experimental rule-based alternative, not a trained model. The active method is displayed above the filters, and scores are recomputed so we can compare how different transparent rules affect the same feedback."},
    {"action": "closing", "narration": "Campus Pulse AI turns comments into service summaries, aspect findings, trends, and a human review queue. Its value is a clear, inspectable workflow that helps identify topics needing attention. Its present limits include unfamiliar slang, multilingual comments, hidden sarcasm, and the use of synthetic teaching data. Future work would collect consented real feedback, label a larger independent evaluation set, improve phrase coverage, and refine rules based on reviewed mistakes."},
]


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def duration(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_app(port, process):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Streamlit exited before becoming ready. See demo_build/streamlit.log")
        try:
            with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Streamlit did not respond within 90 seconds")


async def make_voice_clips(build):
    import edge_tts
    clips = []
    for index, step in enumerate(STORYBOARD):
        path = build / f"voice_{index:02d}.mp3"
        stamp = build / f"voice_{index:02d}.sha256"
        key = hashlib.sha256(f"{VOICE}\n{RATE}\n{step['narration']}".encode("utf-8")).hexdigest()
        if not path.exists() or not stamp.exists() or stamp.read_text(encoding="ascii") != key or duration(path) <= 1:
            for attempt in range(4):
                try:
                    await edge_tts.Communicate(step["narration"], VOICE, rate=RATE).save(str(path))
                    if duration(path) > 1:
                        stamp.write_text(key, encoding="ascii")
                        break
                except Exception as error:
                    if attempt == 3:
                        raise RuntimeError(f"Voice generation failed at step {index + 1}: {error}") from error
                    await asyncio.sleep(2 * (attempt + 1))
        clips.append((path, duration(path)))
        print(f"Voice {index + 1}/{len(STORYBOARD)}: {clips[-1][1]:.1f}s", flush=True)
    return clips


def tab(page, name):
    target = page.get_by_role("tab", name=re.compile(re.escape(name), re.I))
    target.click(timeout=30000)
    page.wait_for_timeout(1100)
    page.evaluate("window.scrollTo({top: 250, behavior: 'smooth'})")
    page.wait_for_timeout(700)


def sidebar_widget(page, label):
    return page.locator('[data-testid="stSidebar"]').get_by_label(label, exact=True)


def perform(page, action):
    if action.startswith("tab:"):
        tab(page, action.split(":", 1)[1])
    elif action == "intro":
        page.evaluate("window.scrollTo(0, 0)")
    elif action == "sidebar":
        sidebar_widget(page, "Student comment").scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        sidebar_widget(page, "Choose analysis method").scroll_into_view_if_needed()
        page.wait_for_timeout(900)
        sidebar_widget(page, "Student comment").scroll_into_view_if_needed()
    elif action == "add_comment":
        sidebar_widget(page, "Student comment").fill("The bus is late, but the driver is polite.")
        page.get_by_role("button", name="Analyze & add feedback").click()
        page.get_by_text("Active scoring method:").wait_for(timeout=45000)
        tab(page, "Feedback explorer")
    elif action == "search_comment":
        page.get_by_placeholder("Search for Wi-Fi, late bus, clean...").fill("The bus is late, but the driver is polite.")
        page.wait_for_timeout(1200)
        tab(page, "Feedback explorer")
    elif action == "switch_algorithm":
        page.get_by_placeholder("Search for Wi-Fi, late bus, clean...").fill("")
        sidebar_widget(page, "Choose analysis method").click()
        page.get_by_role("option", name="Campus Context Score (experimental)").click(timeout=30000)
        page.get_by_text("Active scoring method: Campus Context Score (experimental)").wait_for(timeout=45000)
        tab(page, "Comparison")
    elif action == "closing":
        tab(page, "Overview")
        page.evaluate("window.scrollTo(0, 0)")
    else:
        raise ValueError(action)


def record(build, port, clips):
    from playwright.sync_api import sync_playwright
    video_dir = build / "video"
    video_dir.mkdir(exist_ok=True)
    timing = []
    with sync_playwright() as playwright:
        installed = next((candidate for candidate in CHROME_CANDIDATES if candidate.exists()), None)
        browser = playwright.chromium.launch(headless=True, executable_path=str(installed) if installed else None)
        context = browser.new_context(viewport=SIZE, device_scale_factor=1, record_video_dir=str(video_dir), record_video_size=SIZE)
        page = context.new_page()
        started = time.monotonic()
        page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded", timeout=60000)
        page.get_by_role("tab", name=re.compile("Overview", re.I)).wait_for(timeout=60000)
        page.wait_for_timeout(1600)
        for index, (step, (_, seconds)) in enumerate(zip(STORYBOARD, clips)):
            for attempt in range(3):
                try:
                    perform(page, step["action"])
                    break
                except Exception as error:
                    if attempt == 2:
                        page.screenshot(path=str(build / f"failed_{index:02d}.png"), full_page=True)
                        raise RuntimeError(f"Action {index + 1} ({step['action']}) failed: {error}") from error
                    page.wait_for_timeout(1500 * (attempt + 1))
            offset = time.monotonic() - started
            timing.append({"index": index, "action": step["action"], "start": round(offset, 3), "audio_duration": round(seconds, 3)})
            print(f"Recording {index + 1}/{len(STORYBOARD)}: {step['action']} at {offset:.1f}s", flush=True)
            page.wait_for_timeout(int((seconds + 0.5) * 1000))
        page.wait_for_timeout(1300)
        video = page.video
        context.close()
        source = Path(video.path())
        browser.close()
    (build / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return source, timing


def merge(build, source, clips, timing):
    inputs = []
    filters = []
    for index, (clip, _) in enumerate(clips):
        inputs.extend(["-i", str(clip)])
        delay_ms = round(timing[index]["start"] * 1000)
        filters.append(f"[{index}:a]adelay={delay_ms}|{delay_ms},aresample=async=1[a{index}]")
    joined = "".join(f"[a{i}]" for i in range(len(clips)))
    filters.append(f"{joined}amix=inputs={len(clips)}:duration=longest:dropout_transition=0,apad[aout]")
    audio = build / "aligned_audio.m4a"
    command(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "[aout]", "-t", str(duration(source)), "-c:a", "aac", "-b:a", "192k", str(audio)])
    command(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(OUTPUT)])
    video_seconds = duration(OUTPUT)
    speech_end = max(t["start"] + t["audio_duration"] for t in timing)
    if video_seconds < speech_end - 0.15:
        raise RuntimeError(f"Final speech ends at {speech_end:.1f}s but video ends at {video_seconds:.1f}s")
    print(f"DONE: {OUTPUT} | {video_seconds:.1f}s | {VOICE}")


def main():
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RuntimeError(f"{executable} is missing. On Windows install FFmpeg with: winget install Gyan.FFmpeg")
    try:
        import playwright  # noqa: F401
        import edge_tts  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install dependencies first: python -m pip install -r requirements.txt playwright edge-tts && python -m playwright install chromium") from error
    build = ROOT / "demo_build"
    if build.resolve().parent != ROOT.resolve():
        raise RuntimeError("Unexpected build directory; refusing to remove it")
    build.mkdir(exist_ok=True)
    port = free_port()
    log = (build / "streamlit.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless=true", f"--server.port={port}", "--server.address=127.0.0.1"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_app(port, process)
        clips = asyncio.run(make_voice_clips(build))
        source, timing = record(build, port, clips)
        merge(build, source, clips, timing)
    finally:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


if __name__ == "__main__":
    main()
