"""Unit tests for doodle_visualizer.py"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from doodle_visualizer import (
    LineMorph,
    SequenceBreakdown,
    build_doodle_sequence,
    create_preview_frame,
    parse_lyrics_to_morph_sequence,
)


class TestDoodleVisualizer(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_requirements_file_exists(self) -> None:
        self.assertTrue(os.path.exists("requirements.txt"))
        with open("requirements.txt", encoding="utf-8") as f:
            content = f.read()
        for pkg in ("google-genai", "pillow", "moviepy", "pydantic"):
            self.assertIn(pkg, content)

    def test_schema_instantiation(self) -> None:
        item = LineMorph(
            line_index=0,
            spoken_text="grab a thesaurus",
            kinetic_word="THESAURUS",
            condensed_object="dinosaur",
            morph_action="Melt letters into dinosaur scales",
            visual_description="B&W drawing of dinosaur with lettering THESAURUS",
        )
        self.assertEqual(item.line_index, 0)
        self.assertEqual(item.kinetic_word, "THESAURUS")
        sequence = SequenceBreakdown(items=[item])
        self.assertEqual(len(sequence.items), 1)

    def test_parse_lyrics_missing_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                parse_lyrics_to_morph_sequence("grab a thesaurus")

    def test_create_preview_frame(self) -> None:
        item = LineMorph(
            line_index=99,
            spoken_text="test line for doodle frame rendering",
            kinetic_word="TEST",
            condensed_object="test_object",
            morph_action="Morph action description for continuous B&W transformation",
            visual_description="Visual description",
        )
        frames_dir = os.path.join(self.test_dir, "frames")
        path = create_preview_frame(item, frames_dir=frames_dir)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(Path(path).name, "frame_099.png")

    @patch("doodle_visualizer.genai.Client")
    def test_parse_lyrics_with_gemini(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "items": [
                    {
                        "line_index": 0,
                        "spoken_text": "grab a thesaurus",
                        "kinetic_word": "THESAURUS",
                        "condensed_object": "STEGOSAURUS",
                        "morph_action": "Letters morph to dinosaur",
                        "visual_description": "B&W doodle of dinosaur",
                    }
                ]
            }
        )
        mock_client.models.generate_content.return_value = mock_response

        items = parse_lyrics_to_morph_sequence("grab a thesaurus", api_key="fake_key")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].condensed_object, "STEGOSAURUS")
        mock_client_cls.assert_called_once()

    @patch("doodle_visualizer.parse_lyrics_to_morph_sequence")
    def test_build_doodle_sequence(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = [
            LineMorph(
                line_index=0,
                spoken_text="grab a thesaurus",
                kinetic_word="THESAURUS",
                condensed_object="STEGOSAURUS",
                morph_action="Morph 1",
                visual_description="Desc 1",
            ),
            LineMorph(
                line_index=1,
                spoken_text="get that vocab maxed",
                kinetic_word="MAXED",
                condensed_object="BATTERY METER",
                morph_action="Morph 2",
                visual_description="Desc 2",
            ),
        ]

        sample = "grab a thesaurus\nget that vocab maxed"
        out_mp4 = os.path.join(self.test_dir, "test_output.mp4")
        frames_dir = os.path.join(self.test_dir, "frames")
        prompts = os.path.join(self.test_dir, "doodle_prompts.json")

        sequence = build_doodle_sequence(
            sample,
            sec_per_line=0.5,
            output_mp4=out_mp4,
            frames_dir=frames_dir,
            prompts_path=prompts,
        )
        self.assertEqual(len(sequence), 2)
        self.assertTrue(os.path.exists(out_mp4))
        self.assertTrue(os.path.exists(prompts))

        with open(prompts, encoding="utf-8") as f:
            cues = json.load(f)
        self.assertEqual(len(cues), 2)


if __name__ == "__main__":
    unittest.main()
