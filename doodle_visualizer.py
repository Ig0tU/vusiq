"""doodle_visualizer.py
Cheap / fast / accessible metamorphic B&W doodle engine.
Uses gemini-3.5-flash-lite for semantic deconstruction → continuous topology chains.
Outputs:
  - local preview .mp4 (Pillow + MoviePy)
  - doodle_prompts.json (ready for Google Flow / Veo / any chain video API)
"""

import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, concatenate_videoclips

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ────────────────────────────────────────────────
# Structured output schema (forced by Gemini)
# ────────────────────────────────────────────────
class LineMorph(BaseModel):
    line_index: int
    spoken_text: str = Field(description="Exact original line / phrase")
    kinetic_word: str = Field(description="Single key word or short phrase rendered as literal kinetic word-art")
    condensed_object: str = Field(description="EXACTLY ONE literal black-and-white object that captures the full subtext / entendre of the line")
    morph_action: str = Field(description="How the previous object's geometry continuously deforms into this object (no hard cuts)")
    visual_description: str = Field(description="Strict B&W ink description: how the object and the kinetic word interact")

class SequenceBreakdown(BaseModel):
    items: List[LineMorph]


def mock_parse_lyrics(raw_text: str) -> List[LineMorph]:
    """Fallback parser for local testing when GEMINI_API_KEY is not set."""
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    items = []
    prev_obj = "initial continuous line"
    for idx, line in enumerate(lines):
        words = line.split()
        kw = words[-1] if words else "word"
        obj = f"doodle_{kw}"
        items.append(
            LineMorph(
                line_index=idx,
                spoken_text=line,
                kinetic_word=kw.upper(),
                condensed_object=obj,
                morph_action=f"Continuous line morphs from {prev_obj} into {obj}",
                visual_description=f"Black ink drawing of {obj} with kinetic lettering '{kw.upper()}'",
            )
        )
        prev_obj = obj
    return items


def parse_lyrics_to_morph_sequence(raw_text: str) -> List[LineMorph]:
    """gemini-3.5-flash-lite → chained B&W object + word-art morphs."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not HAS_GENAI:
        print("[Notice] GEMINI_API_KEY not set or google-genai unavailable. Using fallback parser.")
        return mock_parse_lyrics(raw_text)

    client = genai.Client()

    system_instruction = (
        "You are a kinetic visual lyric engineer.\n"
        "Rules (strict):\n"
        "1. Split the input into sequential lines / meaningful phrases.\n"
        "2. For EVERY line, condense the ENTIRE meaning + subtext + double entendre into EXACTLY ONE literal black-and-white object.\n"
        "3. Pair that object with literal B&W kinetic word-art of the most important word/phrase.\n"
        "4. The morph_action must describe a continuous, unbroken line deformation from the previous object's geometry into the new one.\n"
        "5. Aesthetic is pure minimalist black ink linework on pure white paper. No color, no shading, no 3-D, no photorealism.\n"
        "6. Return only valid JSON matching the SequenceBreakdown schema."
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=f"Process this into sequential B&W morph units:\n\n{raw_text}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=SequenceBreakdown,
            temperature=0.15,
        ),
    )

    data = json.loads(response.text)
    return [LineMorph(**item) for item in data["items"]]


def create_preview_frame(item: LineMorph, width: int = 1080, height: int = 1920) -> str:
    """Simple 9:16 B&W preview frame (no fancy fonts required)."""
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    margin = 80
    draw.rectangle([margin, margin, width - margin, height - margin], outline="black", width=6)

    # Header
    draw.text((margin + 40, margin + 40), f"LINE #{item.line_index + 1}", fill="black", font=font)
    draw.text((margin + 40, margin + 90), f'"{item.spoken_text}"', fill="black", font=font)

    # Center content
    cy = height // 2
    draw.text((width // 2 - 200, cy - 140), f"[ OBJECT ]", fill="black", font=font)
    draw.text((width // 2 - 200, cy - 100), item.condensed_object.upper(), fill="black", font=font)
    draw.text((width // 2 - 200, cy - 20), f"[ WORD-ART ]", fill="black", font=font)
    draw.text((width // 2 - 200, cy + 20), item.kinetic_word.upper(), fill="black", font=font)
    draw.text((width // 2 - 280, cy + 100), f"Morph: {item.morph_action[:70]}...", fill="black", font=font)

    os.makedirs("frames", exist_ok=True)
    path = f"frames/frame_{item.line_index:03d}.png"
    img.save(path)
    return path


def build_doodle_sequence(
    raw_text: str,
    sec_per_line: float = 2.5,
    output_mp4: str = "output_preview.mp4",
    style_override: Optional[str] = None,
) -> List[LineMorph]:
    print("[1/3] Parsing with gemini-3.5-flash-lite …")
    sequence = parse_lyrics_to_morph_sequence(raw_text)

    print(f"[2/3] {len(sequence)} morph units generated. Rendering preview frames …")
    clips = []
    for item in sequence:
        print(f"  → L{item.line_index}: “{item.spoken_text}”  →  [{item.condensed_object}] + “{item.kinetic_word}”")
        frame = create_preview_frame(item)
        clips.append(ImageClip(frame).with_duration(sec_per_line))

    print("[3/3] Compiling local preview video …")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_mp4, fps=24, logger=None)
    print(f"\nPreview → {output_mp4}")

    # Export the exact chain for any downstream generative video tool
    cues = [s.model_dump() for s in sequence]
    with open("doodle_prompts.json", "w") as f:
        json.dump(cues, f, indent=2)
    print("Chained prompts → doodle_prompts.json")

    return sequence


if __name__ == "__main__":
    sample = """
grab a thesaurus,
get that vocab maxed..
get a pen and a notepad
find a dope ass track..

jayz said it best when he said
googles your friend bruh..
invest in your growth.
you cant be frugal and win, duhhhh...
"""
    build_doodle_sequence(sample.strip())
