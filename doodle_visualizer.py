"""doodle_visualizer.py
Cheap / fast / accessible metamorphic B&W doodle engine.
Uses gemini-3.5-flash-lite for semantic deconstruction → continuous topology chains.
Outputs:
  - local preview .mp4 (Pillow + MoviePy)
  - doodle_prompts.json (ready for Google Flow / Veo / any chain video API)
"""

import os
import json
import textwrap
from typing import List, Optional
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, concatenate_videoclips

from google import genai
from google.genai import types


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


def parse_lyrics_to_morph_sequence(raw_text: str) -> List[LineMorph]:
    """gemini-3.5-flash-lite → chained B&W object + word-art morphs."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required to run gemini-3.5-flash-lite.")

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
    """Renders a high-quality vertical (9:16) black-and-white doodle preview frame."""
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    margin = 80
    # Double outer border for crisp notebook/doodle frame
    draw.rectangle([margin, margin, width - margin, height - margin], outline="black", width=6)
    draw.rectangle([margin + 12, margin + 12, width - margin - 12, height - margin - 12], outline="black", width=2)

    # Header Card
    header_title = f"BAR #{item.line_index + 1:02d}  |  DOODLE METAMORPHIC ENGINE"
    draw.text((margin + 40, margin + 40), header_title, fill="black", font=font)

    # Wrapped spoken text
    draw.text((margin + 40, margin + 80), "LYRIC LINE:", fill="black", font=font)
    wrapped_lyrics = textwrap.wrap(f'"{item.spoken_text}"', width=65)
    line_y = margin + 110
    for w_line in wrapped_lyrics:
        draw.text((margin + 60, line_y), w_line, fill="black", font=font)
        line_y += 30

    # Center Visual Focus Box
    box_top = line_y + 40
    box_bottom = height - margin - 220
    box_left = margin + 40
    box_right = width - margin - 40
    draw.rectangle([box_left, box_top, box_right, box_bottom], outline="black", width=4)

    cy = (box_top + box_bottom) // 2

    # Visual Object Badge
    draw.text((width // 2 - 220, cy - 180), "┌────────────────────────────────────────┐", fill="black", font=font)
    draw.text((width // 2 - 220, cy - 150), f"  OBJECT:  {item.condensed_object.upper()}", fill="black", font=font)
    draw.text((width // 2 - 220, cy - 120), "└────────────────────────────────────────┘", fill="black", font=font)

    # Word-Art Badge
    draw.text((width // 2 - 220, cy - 40), "┌────────────────────────────────────────┐", fill="black", font=font)
    draw.text((width // 2 - 220, cy - 10), f"  KINETIC WORD-ART:  {item.kinetic_word.upper()}", fill="black", font=font)
    draw.text((width // 2 - 220, cy + 20), "└────────────────────────────────────────┘", fill="black", font=font)

    # Morph Action Section
    morph_y = box_bottom + 30
    draw.text((margin + 40, morph_y), "CONTINUOUS TOPOLOGY TRANSFORMATION:", fill="black", font=font)
    wrapped_morph = textwrap.wrap(item.morph_action, width=70)
    for idx, m_line in enumerate(wrapped_morph):
        draw.text((margin + 60, morph_y + 30 + (idx * 25)), f"• {m_line}", fill="black", font=font)

    os.makedirs("frames", exist_ok=True)
    path = f"frames/frame_{item.line_index:03d}.png"
    img.save(path)
    return path


def build_doodle_sequence(
    raw_text: str = COMPLETE_SAMPLE_LYRICS,
    sec_per_line: float = 1.0,
    output_mp4: str = "output_preview.mp4",
    style_override: Optional[str] = None,
) -> List[LineMorph]:
    print("[1/3] Deconstructing lyrics into continuous B&W doodle morph units via Gemini 3.5 Flash Lite...")
    sequence = parse_lyrics_to_morph_sequence(raw_text)

    print(f"[2/3] {len(sequence)} morph units generated. Rendering 9:16 preview frames...")
    clips = []
    for item in sequence:
        print(f"  → L{item.line_index:02d}: “{item.spoken_text[:35]}...”  →  [{item.condensed_object}] + “{item.kinetic_word}”")
        frame = create_preview_frame(item)
        clips.append(ImageClip(frame).with_duration(sec_per_line))

    print("[3/3] Compiling local preview video...")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_mp4, fps=24, logger=None)
    print(f"\nPreview → {output_mp4}")

    # Export the exact prompt chain for Google Flow / Veo / downstream generative APIs
    cues = [s.model_dump() for s in sequence]
    with open("doodle_prompts.json", "w") as f:
        json.dump(cues, f, indent=2)
    print("Chained prompts → doodle_prompts.json")

    return sequence


if __name__ == "__main__":
    build_doodle_sequence(COMPLETE_SAMPLE_LYRICS.strip())
