"""test_doodle_visualizer.py
Unit tests for doodle_visualizer.py
"""

import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from doodle_visualizer import (
    LineMorph,
    SequenceBreakdown,
    mock_parse_lyrics,
    parse_lyrics_to_morph_sequence,
    create_preview_frame,
    build_doodle_sequence,
)


class TestDoodleVisualizer(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_schema_instantiation(self):
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

    def test_mock_parse_lyrics(self):
        text = "line one\nline two"
        items = mock_parse_lyrics(text)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].line_index, 0)
        self.assertEqual(items[0].spoken_text, "line one")
        self.assertEqual(items[1].line_index, 1)

    def test_create_preview_frame(self):
        item = LineMorph(
            line_index=99,
            spoken_text="test line",
            kinetic_word="TEST",
            condensed_object="test_object",
            morph_action="Morph action description",
            visual_description="Visual description",
        )
        path = create_preview_frame(item)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(path, "frames/frame_099.png")
        if os.path.exists(path):
            os.remove(path)

    @patch("doodle_visualizer.genai.Client")
    def test_parse_lyrics_with_gemini(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "items": [
                {
                    "line_index": 0,
                    "spoken_text": "grab a thesaurus",
                    "kinetic_word": "THESAURUS",
                    "condensed_object": "dinosaur",
                    "morph_action": "Letters morph to dinosaur",
                    "visual_description": "B&W doodle of dinosaur",
                }
            ]
        })
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            items = parse_lyrics_to_morph_sequence("grab a thesaurus")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].condensed_object, "dinosaur")

    def test_build_doodle_sequence(self):
        sample = "grab a thesaurus\nget that vocab maxed"
        out_mp4 = os.path.join(self.test_dir, "test_output.mp4")

        sequence = build_doodle_sequence(sample, sec_per_line=1.0, output_mp4=out_mp4)
        self.assertEqual(len(sequence), 2)
        self.assertTrue(os.path.exists(out_mp4))
        self.assertTrue(os.path.exists("doodle_prompts.json"))

        with open("doodle_prompts.json") as f:
            cues = json.load(f)
            self.assertEqual(len(cues), 2)

        if os.path.exists("doodle_prompts.json"):
            os.remove("doodle_prompts.json")
        if os.path.exists("frames"):
            shutil.rmtree("frames", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
