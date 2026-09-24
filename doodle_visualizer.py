"""VUSIQ — metamorphic B&W doodle engine.

Deconstructs lyrics into a continuous topology chain:
one literal ink object + one kinetic word-art unit per phrase.

Outputs
-------
- 9:16 preview frames (Pillow)
- local preview .mp4 (MoviePy)
- doodle_prompts.json (ready for Flow / Veo / any chain video API)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, concatenate_videoclips

from google import genai
from google.genai import types


class LineMorph(BaseModel):
    line_index: int
    spoken_text: str = Field(description="Exact original line / phrase")
    kinetic_word: str = Field(
        description="Single key word or short phrase rendered as literal kinetic word-art"
    )
    condensed_object: str = Field(
        description="EXACTLY ONE literal black-and-white object that captures the full subtext / entendre of the line"
    )
    morph_action: str = Field(
        description="How the previous object's geometry continuously deforms into this object (no hard cuts)"
    )
    visual_description: str = Field(
        description="Strict B&W ink description: how the object and the kinetic word interact"
    )


class SequenceBreakdown(BaseModel):
    items: List[LineMorph]


DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920

SYSTEM_INSTRUCTION = (
    "You are a kinetic visual lyric engineer.\n"
    "Rules (strict):\n"
    "1. Split the input into sequential lines / meaningful phrases.\n"
    "2. For EVERY line, condense the ENTIRE meaning + subtext + double entendre "
    "into EXACTLY ONE literal black-and-white object.\n"
    "3. Pair that object with literal B&W kinetic word-art of the most important word/phrase.\n"
    "4. The morph_action must describe a continuous, unbroken line deformation "
    "from the previous object's geometry into the new one. First item: describe how the object is born from a single ink stroke.\n"
    "5. Aesthetic is pure minimalist black ink linework on pure white paper. "
    "No color, no shading, no 3-D, no photorealism.\n"
    "6. Return only valid JSON matching the SequenceBreakdown schema."
)

COMPLETE_SAMPLE_LYRICS = """grab a thesaurus,
get that vocab maxed..
get a pen and a notepad
find a dope ass track..

jayz said it best when he said

googles your friend bruh..

invest in your growth.

you cant be frugal and win, duhhhh...

you might think up some killer dope hits ...

at pinnacle tip,

but when you go spit it

it really dont click,

Ig0tU

the feeling youll get

when you get off that all of that silly bullshit

and find your way onto some syllable shit

man wait.. before we move for words,

your choice -- we can have beef or we move forwards

same silly bullshit game we played 2 bars back
---
same syllable shit. game. we played 2 bars back to back to... yeah...

you getting it yet?

heres where i should mention phonetics

intention-connections,

that switch on you the instant you get it.

yeah , i hyu on insta-joo-get it?

metaphors and similes, hat tips to the legends



by the way i call em em-dees.. officially switched it since
these slant rhymers dont even know who emily dickinson is ..

relativity trees grow, when they they listen it splits

the pool of who gets it slim as the pickings can get...

lets go to barchitechture...

and how you build 1 verse can mark a sector..

then lots a sectors

now you gotta go market sectors..

thats not repetitive,
have em take what gets said with it,

now you making 'em art collectors...

the vibe continues.. the pocket differs. .

like the topics..

what i can then do'is.. pick whats delivered..

whos signing off it..

breaking it down so.. ev-er-y sounds known..

and let it build in full as syllables fill the gaps showing

and then thats rolls in to faster flowing

its all a matter of mathematics, though  ask beethoven

fun fact, the dude was deaf, still he was mad composing..

producers to this day use still use those same classical hints

and thats why only time can tell us, what last, and wont stick...

and any body can have a moment, you have to own it..

class is over i have to go get rap back to dope shit


a fraction of us are actually potent you have to know this..

before you know how to rap."""


_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def parse_lyrics_to_morph_sequence(
    raw_text: str,
    *,
    model: str = DEFAULT_MODEL,
    style_override: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[LineMorph]:
    """gemini structured output → chained B&W object + word-art morphs."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is required. Export it or pass --api-key."
        )

    client = genai.Client(api_key=key)

    instruction = SYSTEM_INSTRUCTION
    if style_override:
        instruction += f"\n7. Additional style constraint: {style_override.strip()}"

    response = client.models.generate_content(
        model=model,
        contents=f"Process this into sequential B&W morph units:\n\n{raw_text}",
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            response_mime_type="application/json",
            response_schema=SequenceBreakdown,
            temperature=0.15,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        payload = json.loads(response.text)
        breakdown = SequenceBreakdown.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Gemini response did not match schema: {exc}") from exc

    return breakdown.items


def create_preview_frame(
    item: LineMorph,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    frames_dir: str | os.PathLike[str] = "frames",
) -> str:
    """Render a 9:16 black-and-white storyboard frame."""
    img = Image.new("RGB", (width, height), color="#FAFAF8")
    draw = ImageDraw.Draw(img)

    font_xs = _load_font(22)
    font_sm = _load_font(28)
    font_md = _load_font(36)
    font_lg = _load_font(52)
    font_xl = _load_font(68)

    margin = 72
    ink = "#0B0B0C"

    draw.rectangle([margin, margin, width - margin, height - margin], outline=ink, width=6)
    draw.rectangle(
        [margin + 14, margin + 14, width - margin - 14, height - margin - 14],
        outline=ink,
        width=2,
    )

    header = f"VUSIQ  ·  BAR {item.line_index + 1:02d}"
    draw.text((margin + 40, margin + 36), header, fill=ink, font=font_sm)

    draw.text((margin + 40, margin + 88), "LYRIC", fill=ink, font=font_xs)
    wrapped_lyrics = textwrap.wrap(f'"{item.spoken_text}"', width=42)
    line_y = margin + 118
    for w_line in wrapped_lyrics[:6]:
        draw.text((margin + 40, line_y), w_line, fill=ink, font=font_md)
        line_y += 44

    box_top = line_y + 36
    box_bottom = height - margin - 280
    box_left = margin + 36
    box_right = width - margin - 36
    draw.rectangle([box_left, box_top, box_right, box_bottom], outline=ink, width=3)

    cy = (box_top + box_bottom) // 2
    cx = width // 2

    draw.text((cx, cy - 140), "OBJECT", fill=ink, font=font_xs, anchor="mm")
    object_lines = textwrap.wrap(item.condensed_object.upper(), width=22) or [""]
    oy = cy - 80
    for line in object_lines[:3]:
        draw.text((cx, oy), line, fill=ink, font=font_xl, anchor="mm")
        oy += 76

    draw.line([(box_left + 48, cy + 40), (box_right - 48, cy + 40)], fill=ink, width=2)

    draw.text((cx, cy + 80), "KINETIC WORD", fill=ink, font=font_xs, anchor="mm")
    word_lines = textwrap.wrap(item.kinetic_word.upper(), width=24) or [""]
    wy = cy + 130
    for line in word_lines[:2]:
        draw.text((cx, wy), line, fill=ink, font=font_lg, anchor="mm")
        wy += 60

    morph_y = box_bottom + 28
    draw.text((margin + 40, morph_y), "CONTINUOUS MORPH", fill=ink, font=font_xs)
    wrapped_morph = textwrap.wrap(item.morph_action, width=48)
    for idx, m_line in enumerate(wrapped_morph[:5]):
        draw.text((margin + 40, morph_y + 32 + idx * 32), m_line, fill=ink, font=font_sm)

    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)
    path = frames_path / f"frame_{item.line_index:03d}.png"
    img.save(path, format="PNG")
    return str(path)


def build_doodle_sequence(
    raw_text: str = COMPLETE_SAMPLE_LYRICS,
    sec_per_line: float = 1.0,
    output_mp4: str = "output_preview.mp4",
    style_override: Optional[str] = None,
    *,
    frames_dir: str = "frames",
    prompts_path: str = "doodle_prompts.json",
    model: str = DEFAULT_MODEL,
    write_video: bool = True,
) -> List[LineMorph]:
    print("[1/3] Deconstructing lyrics into B&W morph units…")
    sequence = parse_lyrics_to_morph_sequence(
        raw_text, model=model, style_override=style_override
    )

    print(f"[2/3] {len(sequence)} units. Rendering 9:16 frames…")
    clips = []
    for item in sequence:
        snippet = item.spoken_text.replace("\n", " ")[:36]
        print(
            f"  → L{item.line_index:02d}: “{snippet}…”  "
            f"[{item.condensed_object}] + “{item.kinetic_word}”"
        )
        frame = create_preview_frame(item, frames_dir=frames_dir)
        if write_video:
            clips.append(ImageClip(frame).with_duration(sec_per_line))

    if write_video:
        if not clips:
            raise RuntimeError("No frames to compile.")
        print("[3/3] Compiling preview video…")
        final = concatenate_videoclips(clips, method="compose")
        Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
        final.write_videofile(output_mp4, fps=24, logger=None)
        print(f"Preview → {output_mp4}")
    else:
        print("[3/3] Skipping video compile.")

    cues = [s.model_dump() for s in sequence]
    Path(prompts_path).parent.mkdir(parents=True, exist_ok=True)
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(cues, f, indent=2, ensure_ascii=False)
    print(f"Chained prompts → {prompts_path}")

    return sequence


def _read_lyrics(path: Optional[str], inline: Optional[str]) -> str:
    if inline:
        return inline
    if path:
        return Path(path).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return COMPLETE_SAMPLE_LYRICS


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vusiq",
        description="Metamorphic B&W lyric doodle engine.",
    )
    parser.add_argument("--lyrics", "-l", help="Path to a lyrics .txt file")
    parser.add_argument("--text", "-t", help="Inline lyrics text")
    parser.add_argument("--out", "-o", default="output_preview.mp4", help="Preview mp4 path")
    parser.add_argument("--prompts", default="doodle_prompts.json", help="JSON cue export path")
    parser.add_argument("--frames-dir", default="frames", help="Directory for PNG frames")
    parser.add_argument("--sec", type=float, default=1.0, help="Seconds per bar in the preview")
    parser.add_argument("--style", default=None, help="Extra style constraint for Gemini")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id")
    parser.add_argument("--no-video", action="store_true", help="Skip mp4 compile")
    args = parser.parse_args(argv)

    lyrics = _read_lyrics(args.lyrics, args.text).strip()
    if not lyrics:
        print("No lyrics provided.", file=sys.stderr)
        return 2

    build_doodle_sequence(
        lyrics,
        sec_per_line=args.sec,
        output_mp4=args.out,
        style_override=args.style,
        frames_dir=args.frames_dir,
        prompts_path=args.prompts,
        model=args.model,
        write_video=not args.no_video,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
