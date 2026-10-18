import unittest

import pandas as pd

from src.session import apply_change, clear_dataset, restore_original, start_dataset, undo_change


class SessionTests(unittest.TestCase):
    def test_clean_undo_and_restore_are_independent(self):
        state = {}
        original = pd.DataFrame({"x": [1, 1, 2]})
        start_dataset(state, original, {}, "first")
        apply_change(state, state["df"].drop_duplicates(), "duplicates")
        self.assertEqual(len(state["df"]), 2)
        pd.testing.assert_frame_equal(state["original_df"], original)
        undo_change(state)
        pd.testing.assert_frame_equal(state["df"], original)
        apply_change(state, state["df"].iloc[:0], "all rows")
        restore_original(state)
        pd.testing.assert_frame_equal(state["df"], original)
        self.assertEqual(state["history"], [])

    def test_new_source_discards_old_history(self):
        state = {"unrelated": True}
        start_dataset(state, pd.DataFrame({"x": [1]}), {}, "a")
        apply_change(state, state["df"].iloc[:0], "remove")
        start_dataset(state, pd.DataFrame({"x": [2]}), {}, "b")
        self.assertEqual(state["changes"], [])
        self.assertEqual(state["df"].iloc[0, 0], 2)
        clear_dataset(state)
        self.assertEqual(state, {"unrelated": True})
