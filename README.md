# VUSIQ

**Metamorphic black-and-white lyric doodle engine.**

VUSIQ takes a verse and collapses each line into one literal ink object plus one kinetic word. The objects do not cut — they deform. The result is a storyboard you can watch locally and a prompt chain you can hand to Flow, Veo, or any sequential video model.

Ink on paper. No neon. No candy palette.

## What it does

1. **Deconstruct** — Gemini structured output splits lyrics into sequential morph units.
2. **Condense** — each unit is exactly one object + one word-art phrase.
3. **Chain** — `morph_action` describes continuous topology from the previous geometry.
4. **Preview** — 9:16 storyboard frames compiled to `mp4`.
5. **Export** — `doodle_prompts.json` for downstream generative video.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY=your_key

# bundled sample verse
python doodle_visualizer.py

# your own lyrics
python doodle_visualizer.py --lyrics path/to/verse.txt --out preview.mp4

# storyboard + JSON only
python doodle_visualizer.py --text "grab a thesaurus" --no-video
```

## CLI

```
python doodle_visualizer.py
  --lyrics PATH        lyrics file
  --text STRING        inline lyrics
  --out PATH           preview mp4 (default: output_preview.mp4)
  --prompts PATH       JSON export (default: doodle_prompts.json)
  --frames-dir PATH    PNG frames directory
  --sec FLOAT          seconds per bar (default: 1.0)
  --style STRING       extra aesthetic constraint for Gemini
  --model ID           Gemini model (default: gemini-3.5-flash-lite)
  --no-video           skip mp4 compile
```

Stdin works when no `--lyrics` / `--text` is given and the process is not a TTY:

```bash
cat verse.txt | python doodle_visualizer.py --no-video
```

## Output schema

Each item in `doodle_prompts.json`:

| field | meaning |
| --- | --- |
| `line_index` | order in the chain |
| `spoken_text` | original phrase |
| `kinetic_word` | word-art fragment |
| `condensed_object` | the single ink object |
| `morph_action` | how the last object becomes this one |
| `visual_description` | B&W interaction of object + word |

## Requirements

- Python 3.10+
- `GEMINI_API_KEY`
- FFmpeg on PATH (MoviePy preview compile)

## Tests

```bash
python -m unittest test_doodle_visualizer.py -v
```

Gemini calls are mocked. Frame + video tests write into a temp directory.

## Aesthetic

Near-black ink `#0B0B0C` on paper `#FAFAF8`. Double-rule notebook frame. Vertical 9:16. The preview is a storyboard, not the finished film — the JSON is the artifact you feed a real motion model.

## License

MIT. See [LICENSE](LICENSE).
