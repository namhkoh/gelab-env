# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_04
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6.png
# step_index: 4/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. This script paints the page background,
# status bar, header area, and subtle separators / section card backgrounds
# while avoiding the detected element areas.

w, h = canvas.size

# Overall page background (slightly off-white to match screenshot)
draw.rectangle([(0, 0), (w, h)], fill="#FBFCFD")

# Status bar (top area)
status_h = 92
draw.rectangle([(0, 0), (w, status_h)], fill="#E6E6E6")

# Header area (toolbar) below status bar
header_top = status_h
header_bottom = 184
draw.rectangle([(0, header_top), (w, header_bottom)], fill="#FFFFFF")

# Subtle bottom divider / shadow under header
draw.line([(24, header_bottom), (w - 24, header_bottom)], fill="#ECEFF3", width=1)
draw.rectangle([(0, header_bottom), (w, header_bottom + 2)], fill="#E9EDF2")

# Helper to avoid drawing over detected interactive bands:
# reserved vertical bands to not draw inside (segmented control and bottom apply area)
segmented_top = 2024
segmented_bottom = 2024 + 144  # 2168
apply_top = 2768
apply_bottom = apply_top + 144  # 2912

# Function to draw a horizontal separator but skip if it would intersect reserved bands
def safe_hline(y, x1=36, x2=None, color="#F1F3F6", width=1):
    if x2 is None:
        x2 = w - 36
    # If y lies inside any reserved vertical band (segmented or apply), skip drawing
    if segmented_top <= y <= segmented_bottom or apply_top <= y <= apply_bottom:
        return
    draw.line([(x1, y), (x2, y)], fill=color, width=width)

# Section separators (placed between major groups)
safe_hline(header_bottom + 376)   # ~560
safe_hline(980)                   # after Event type section
safe_hline(1420)                  # after Languages section
safe_hline(1680)                  # after Price / toggle area
safe_hline(1950)                  # spacer above Sort by area

# Subtle section card backgrounds (rounded rectangles) that do not overlap detected interactive areas.
# Categories / chips area background (very subtle rounded inset)
cat_card_top = 300
cat_card_bottom = 560
if not (cat_card_top <= segmented_bottom and cat_card_bottom >= segmented_top):
    draw.rounded_rectangle(
        [(36, cat_card_top), (w - 36, cat_card_bottom)],
        radius=14,
        fill="#FFFFFF",
        outline="#F3F5F8",
        width=1
    )

# Event type card (subtle)
etype_top = 720
etype_bottom = 980
draw.rounded_rectangle(
    [(36, etype_top), (w - 36, etype_bottom)],
    radius=14,
    fill="#FFFFFF",
    outline="#F3F5F8",
    width=1
)

# Languages card (subtle)
lang_top = 1120
lang_bottom = 1420
draw.rounded_rectangle(
    [(36, lang_top), (w - 36, lang_bottom)],
    radius=14,
    fill="#FFFFFF",
    outline="#F3F5F8",
    width=1
)

# Price / toggle area card (subtle)
price_top = 1560
price_bottom = 1740
draw.rounded_rectangle(
    [(36, price_top), (w - 36, price_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline="#F3F5F8",
    width=1
)

# Add a subtle rounded tray/background for the Sort by area but ensure we do not draw inside the segmented control band.
tray_top = 1880
tray_bottom = 2008  # placed above the segmented control; avoid overlap
if tray_bottom <= segmented_top:
    draw.rounded_rectangle(
        [(54, tray_top), (w - 54, tray_bottom)],
        radius=12,
        fill="#FAFBFC",
        outline="#E9EDF2",
        width=1
    )

# Subtle left and right page margins as faint lines to frame content (do not cross reserved bottom apply area)
safe_hline(2400, x1=48, x2=w-48, color="#FFFFFF", width=0)  # placeholder to ensure margins exist but invisible

# small horizontal accent lines under various section headers (where headers would be)
accent_y_positions = [260, 680, 1050, 1250, 1600, 1900]
for y in accent_y_positions:
    safe_hline(y, x1=48, x2=w-48, color="#F7F8FA", width=1)

# Finished drawing layout/background elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1275), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 135)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/18_icon_9.09.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.09"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/19_icon_9.09.png
try:
    _c19 = get_crop(19, 63, 63)
    canvas.paste(_c19, (176, 2), _c19)
except Exception:
    pass
layout["9.09"] = [176, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 99, 65)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 56, 67)
    canvas.paste(_c21, (1317, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1317, 0, 1373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/22_icon_9.09.png
try:
    _c22 = get_crop(22, 57, 65)
    canvas.paste(_c22, (113, 1), _c22)
except Exception:
    pass
layout["9.09"] = [113, 1, 170, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/23_icon_clickable_20.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1729), _c23)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 178, 144)
    canvas.paste(_c24, (1214, 72), _c24)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 59, 62)
    canvas.paste(_c25, (245, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [245, 2, 304, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 54, 61)
    canvas.paste(_c26, (314, 3), _c26)
except Exception:
    pass
layout["icon_26"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/27_text_9.09.png
try:
    _c27 = get_crop(27, 94, 45)
    canvas.paste(_c27, (17, 15), _c27)
except Exception:
    pass
layout["9.09"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_04_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-6/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
