# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_03
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5.png
# step_index: 3/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile Filters page
# Uses provided variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm/font_md/font_lg/font_xl

# Full white background (match dominant color)
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar area (top)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#cfcfcf")  # light grey status bar

# Header / toolbar area beneath status bar
header_y0 = status_h
header_y1 = status_h + 96
draw.rectangle((0, header_y0, 1440, header_y1), fill="#ffffff")
# header bottom divider
draw.line((24, header_y1, 1440 - 24, header_y1), fill="#e6e6ea", width=1)

# Section separators between major groups
separator_color = "#f1f1f6"
separators = [
    520,   # after Categories / "Show all categories"
    964,   # after Event type / "Show all event types"
    1410,  # after Languages / "Show all languages"
    1613,  # after Price title area
    1936,  # above Sort by control
]
for y in separators:
    draw.line((36, y, 1440 - 36, y), fill=separator_color, width=1)

# Light background band behind the Sort-by segmented control (so the segments pasted on top sit on a soft container)
sort_container_y0 = 1988
sort_container_y1 = 2176
draw.rounded_rectangle(
    (36, sort_container_y0, 1440 - 36, sort_container_y1),
    radius=14,
    fill="#f7f6f9",
    outline="#e6e3ea",
)

# Subtle shadow under the sort container to lift it off the page
shadow_y = sort_container_y1 + 6
draw.rectangle((40, shadow_y, 1440 - 40, shadow_y + 3), fill="#efecef")

# Light section card behind the "Only free events" toggle area (gives grouping without drawing the toggle itself)
price_card_y0 = 1560
price_card_y1 = 2050
draw.rounded_rectangle(
    (24, price_card_y0, 1440 - 24, price_card_y1),
    radius=12,
    fill="#ffffff",
    outline=None
)

# Bottom safe area / area behind the Apply filters button
bottom_area_y0 = 2640
draw.rectangle((0, bottom_area_y0, 1440, 2960), fill="#fbfbfd")
# top border for bottom area
draw.line((24, bottom_area_y0, 1440 - 24, bottom_area_y0), fill="#e2dee4", width=2)

# Subtle inset border where the Apply filters button will be pasted (do not draw button itself)
draw.rounded_rectangle(
    (48, 2768 - 16, 1440 - 48, 2768 + 144 + 16),
    radius=10,
    outline="#d3cfd6",
    width=2,
    fill=None
)

# Additional faint vertical guides/columns to match screenshot spacing (purely structural)
# left content margin guide (very faint)
draw.line((36, header_y1 + 8, 36, bottom_area_y0 - 8), fill="#fbfbfb", width=1)
# right content margin guide (very faint)
draw.line((1440 - 36, header_y1 + 8, 1440 - 36, bottom_area_y0 - 8), fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/11_icon_German.png
try:
    _c11 = get_crop(11, 225, 135)
    canvas.paste(_c11, (270, 1275), _c11)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/12_icon_Festival.png
try:
    _c12 = get_crop(12, 219, 144)
    canvas.paste(_c12, (1122, 829), _c12)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/18_icon_9.09.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.09"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/19_icon_9.09.png
try:
    _c19 = get_crop(19, 64, 64)
    canvas.paste(_c19, (176, 1), _c19)
except Exception:
    pass
layout["9.09"] = [176, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 55, 69)
    canvas.paste(_c20, (1319, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 99, 69)
    canvas.paste(_c21, (1211, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/22_icon_9.09.png
try:
    _c22 = get_crop(22, 58, 65)
    canvas.paste(_c22, (112, 1), _c22)
except Exception:
    pass
layout["9.09"] = [112, 1, 170, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 63)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 54, 61)
    canvas.paste(_c24, (314, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/27_text_9.09.png
try:
    _c27 = get_crop(27, 94, 45)
    canvas.paste(_c27, (17, 15), _c27)
except Exception:
    pass
layout["9.09"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_03_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-5/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
