import unittest
from io import BytesIO

import pandas as pd

from src.utilities import load_dataframe


def upload(contents, name):
    stream = BytesIO(contents)
    stream.name = name
    return stream


class LoadingOptionsTests(unittest.TestCase):
    def test_explicit_encoding_separator_and_header(self):
        source = upload("город;число\nМосква;2\nКазань;3".encode("cp1251"), "cities.csv")
        frame, info = load_dataframe(source, encoding="cp1251", separator=";", header="present")
        self.assertIsNotNone(frame, info)
        self.assertEqual(frame.iloc[0, 0], "Москва")
        self.assertEqual(frame.columns.tolist(), ["город", "число"])
        self.assertEqual(info["separator"], ";")

    def test_headerless_text_keeps_first_row(self):
        frame, info = load_dataframe(upload(b"Alice;Paris\nBob;London", "people.csv"),
                                     separator=";", header="absent")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0, 0], "Alice")
        self.assertTrue(info["header_generated"])

    def test_wrong_encoding_is_reported_without_silent_fallback(self):
        frame, error = load_dataframe(upload("Москва".encode("cp1251"), "a.csv"), encoding="utf-8")
        self.assertIsNone(frame)
        self.assertIn("Ошибка", error)

    def test_excel_sheet_selection_and_headerless_numeric_rows(self):
        content = BytesIO()
        with pd.ExcelWriter(content, engine="openpyxl") as writer:
            pd.DataFrame({"x": [99]}).to_excel(writer, sheet_name="first", index=False)
            pd.DataFrame([[1, 2], [3, 4]]).to_excel(writer, sheet_name="second", index=False, header=False)
        frame, info = load_dataframe(upload(content.getvalue(), "book.xlsx"), sheet_name="second", header="absent")
        self.assertEqual(frame.values.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(info["sheet_name"], "second")

    def test_empty_and_invalid_files_return_errors(self):
        for contents, name in [(b"", "empty.csv"), (b"{broken", "bad.json"), (b"abc", "file.txt")]:
            frame, message = load_dataframe(upload(contents, name))
            self.assertIsNone(frame)
            self.assertTrue(message)
