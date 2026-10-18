import unittest

from streamlit.testing.v1 import AppTest


def app_script(contents):
    return f'''
from io import BytesIO
from unittest.mock import patch
import main
source = BytesIO({contents!r})
source.name = "test.csv"
with patch("streamlit.file_uploader", return_value=source):
    main.configure_streamlit()
'''


def button(app, label):
    return next(item for item in app.button if item.label == label)


class AppTests(unittest.TestCase):
    def test_preview_clean_undo_restore_and_report(self):
        app = AppTest.from_string(app_script(b"amount,count\n1,2\n1,2\n3,4\n5,6\n7,8\n9,10\n100,200"), default_timeout=60)
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.slider), 0)
        button(app, "Начать анализ").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["df"]), 7)
        button(app, "Удалить дубликаты: 1 строк").click().run()
        self.assertEqual(len(app.session_state["df"]), 6)
        button(app, "Отменить последнее действие").click().run()
        self.assertEqual(len(app.session_state["df"]), 7)
        button(app, "Удалить дубликаты: 1 строк").click().run()
        button(app, "Вернуть исходные данные").click().run()
        self.assertEqual(len(app.session_state["df"]), 7)
        app.slider(key="if_contamination").set_value(0.2).run()
        button(app, "Подготовить Excel-отчёт").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("download_button")), 1)
        next(item for item in app.checkbox if item.label == "Показать графики").check().run()
        self.assertEqual(len(app.exception), 0)

    def test_text_only_and_missing_values_render(self):
        app = AppTest.from_string(app_script(b"name,city\nAlice,Paris\nBob,\nCarol,Rome"), default_timeout=60)
        app.run()
        button(app, "Начать анализ").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.slider), 0)
        button(app, "Подготовить Excel-отчёт").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("download_button")), 1)
