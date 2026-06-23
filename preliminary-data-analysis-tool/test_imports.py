import sys
sys.path.insert(0, '.')
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Full import test
errors = []
try:
    from utilities import (
        create_image_reportlab, load_dataframe, profile_dataframe,
        detect_outliers_iqr, export_csv, create_list_of_lists,
        set_head_for_df_table, create_title_with_df, draw_boxplot,
        plot_correlation_heatmap, plot_missing_values, guess_csv_params
    )
    print("utilities.py: OK")
except Exception as e:
    errors.append(f"utilities.py: {e}")
    print(f"utilities.py: FAIL - {e}")

try:
    from app import (
        use_pygwalker, count_values_by_column, drop_visualization,
        corr_visaulization, duplicate_search, outliers_iqr_search,
        create_pdf_review
    )
    print("app.py: OK")
except Exception as e:
    errors.append(f"app.py: {e}")
    print(f"app.py: FAIL - {e}")

try:
    from config import TextStyle, get_table_style
    print("config.py: OK")
except Exception as e:
    errors.append(f"config.py: {e}")
    print(f"config.py: FAIL - {e}")

try:
    import main
    print("main.py: OK")
except Exception as e:
    errors.append(f"main.py: {e}")
    print(f"main.py: FAIL - {e}")

if errors:
    print(f"\nERRORS FOUND ({len(errors)}):")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("\nAll imports successful - no errors!")