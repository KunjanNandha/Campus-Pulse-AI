# Campus Pulse AI

Campus Pulse AI is a transparent Streamlit dashboard for turning student
feedback into actionable campus-service insights. It keeps the original
comment visible, explains how each score was produced, and helps reviewers
find recurring issues across services such as Wi-Fi, transport, classrooms,
the canteen, laboratories, cleanliness, campus facilities, and the library.

The project is intentionally rule-based. It does not train or require a
machine-learning classifier, making the scoring pipeline easy to inspect,
test, and adapt for a campus feedback workflow.

## What the project includes

- Interactive Streamlit dashboard with overview, service, feedback, trend,
  evaluation, and review screens.
- Built-in synthetic demonstration data covering all supported services.
- CSV import and analyzed CSV export.
- Automatic service detection with keyword matching.
- VADER sentiment scoring with a campus-specific lexicon.
- Independent Campus Context Score for transparent rule-based comparison.
- Negation, emphasis, punctuation, contrast, emoji, abbreviation, and
  selected slang handling.
- Fine-grained sentiment labels, mixed-opinion detection, and derived stars.
- Aspect-level analysis for service concerns such as speed, price, hygiene,
  punctuality, seating, and equipment.
- Priority action suggestions and early-warning trend detection.
- Sarcasm and low-confidence flags that route difficult comments to human
  review.
- Hand-labelled evaluation examples and a review queue that preserves the
  original algorithm output.
- Automated narrated demo recording with `make_demo.py`.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit dashboard and analysis pipeline |
| `test_app.py` | Focused regression tests for sentiment and aspect rules |
| `make_demo.py` | Creates a narrated browser walkthrough |
| `requirements.txt` | Runtime and test dependencies |
| `.streamlit/config.toml` | Dashboard theme configuration |
| `Poster SMA.png` | Project poster asset |

Generated recordings, browser downloads, caches, and local environments are
excluded through `.gitignore`.

## Run locally

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The dashboard starts with its
synthetic dataset. Use the sidebar to restore the demo data, add a comment,
upload a CSV, select the scoring method, or add temporary lexicon entries.

## CSV input

CSV uploads require a `feedback` column. These columns are optional:

- `service`: one of the supported campus services.
- `timestamp`: a parseable date or timestamp.
- `rating`: an integer from 1 to 5.

Missing or unrecognized services are inferred from whole-word keywords.
Invalid dates use the import time. Imported and manually added comments are
kept only in the current Streamlit session. The Feedback explorer can export
the fully analyzed data as CSV.

## Analysis pipeline

1. Preserve the original comment and create a separate scoring copy.
2. Expand only explicitly supported student abbreviations.
3. Match VADER and campus-specific lexicon entries.
4. Apply the selected scoring method and its transparent context rules.
5. Score clauses separately to identify opposing opinions and aspects.
6. Apply the cautious sarcasm rule only when both a sarcasm cue and a
   concrete complaint are present.
7. Produce sentiment labels, numeric scores, derived stars, flags, charts,
   alerts, and exports.

`VADER + campus lexicon` is the default method. `Campus Context Score` is an
independent experimental scorer with its own negation window, boosters,
capitalization, contrast weighting, punctuation scaling, and emoji values.
Both methods are hand-written teaching rules, not calibrated probabilities.

## Testing

Run the focused regression suite:

```powershell
python -m unittest -v test_app.py
```

The tests cover abbreviation expansion, mixed clause sentiment, aspect
classification, multi-sentence comments, and the sarcasm complaint guard.

## Optional narrated demo

Install the extra demo tools:

```powershell
python -m pip install playwright edge-tts
$env:PLAYWRIGHT_BROWSERS_PATH = (Join-Path (Get-Location) 'demo_browsers')
python -m playwright install ffmpeg
python -m playwright install chromium
```

If FFmpeg is not already available on Windows, install it with:

```powershell
winget install Gyan.FFmpeg
```

Generate the walkthrough:

```powershell
python make_demo.py
```

The output is `demo_narrated.mp4`. Set `DEMO_VOICE` or `DEMO_RATE` before
running the script to customize the narration.

## Limitations

The dictionaries and rules are English-oriented and can misread unseen
sarcasm, slang, multilingual text, distant negation, or very short comments.
Clause splitting and keyword matching can lose context, and synthetic demo
data must not be treated as campus-wide evidence. Alerts and review flags are
prompts for investigation, not confirmed incidents or validated predictions.
Review the original comments before making decisions.

## License

This project is provided for educational and demonstration purposes. Add the
license terms required by your institution or deployment context before
redistributing it.
