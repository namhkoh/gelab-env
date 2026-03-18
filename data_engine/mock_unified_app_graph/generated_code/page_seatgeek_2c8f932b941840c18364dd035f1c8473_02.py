# page_id: page_seatgeek_2c8f932b941840c18364dd035f1c8473_02
# screenshot: 2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5.png
# step_index: 2/8
# task: Open SeatGeek. Search "Beatles Love". Select the soonest upcoming event. Choose 2 tickets and continue. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas.
# Available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill overall background (match app's neutral white)
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Top status bar area (light gray)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#efefef")

# Search bar (rounded) below the status bar
search_left = 40
search_top = 60
search_right = 1400
search_bottom = 180
draw.rounded_rectangle(
    (search_left, search_top, search_right, search_bottom),
    radius=32,
    fill="#f6f6f6",
    outline="#e6e6e6",
    width=1
)

# Subtle divider line immediately under the search bar area
divider_y_1 = search_bottom + 12
draw.line((search_left, divider_y_1, search_right, divider_y_1), fill="#e9e9e9", width=1)

# Section separator between the "Recent searches" list and the suggestions area
# (estimated based on detected items; keep very subtle)
recent_splits_y = 1312
draw.line((24, recent_splits_y, 1440-24, recent_splits_y), fill="#eeeeee", width=1)

# Another faint horizontal rule further up to visually separate header/search area from content
draw.line((24, 312, 1440-24, 312), fill="#f0f0f0", width=1)

# Bottom navigation bar background and top divider/shadow
nav_top = 2792
nav_bottom = 2960
draw.rectangle((0, nav_top, 1440, nav_bottom), fill="#ffffff")
# thin divider/shadow on top of nav bar
draw.line((0, nav_top, 1440, nav_top), fill="#eaeaea", width=2)

# Light grouping background behind the main content area (very subtle)
# This provides a slight off-white canvas for content without overlapping icons/text
content_left = 24
content_top = search_bottom + 40
content_right = 1440 - 24
content_bottom = nav_top - 24
draw.rounded_rectangle(
    (content_left, content_top, content_right, content_bottom),
    radius=8,
    fill="#ffffff",
    outline="#fafafa",
    width=1
)

# Subtle vertical padding indicators (thin lines) to imply item boundaries (non-intrusive)
# These are faint and positioned to avoid overlapping pasted icons and text.
for y in (520, 688, 856, 1024, 1192):
    draw.line((40, y, 1400, y), fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/00_icon_Recent_searches.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 47, 70)
    canvas.paste(_c1, (1153, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/02_icon_Wicked.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 639), _c2)
except Exception:
    pass
layout["Wicked"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/03_icon_The_Phantom_of_the_Opera.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 471), _c3)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 66, 62)
    canvas.paste(_c5, (242, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [242, 3, 308, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 96, 69)
    canvas.paste(_c6, (1217, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1217, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/08_icon_The_Phantom_of_the_Opera.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 639), _c8)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/10_icon_Just_Announced_by_My_Performers.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1688), _c10)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/11_icon_Boston_Celtics.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 807), _c11)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/13_icon_5.06_my.png
try:
    _c13 = get_crop(13, 168, 144)
    canvas.paste(_c13, (48, 120), _c13)
except Exception:
    pass
layout["5.06_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/14_icon_Suggestions.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 1143), _c14)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/15_icon_Account.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (1152, 2792), _c15)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/16_icon_5.06_my.png
try:
    _c16 = get_crop(16, 47, 63)
    canvas.paste(_c16, (186, 1), _c16)
except Exception:
    pass
layout["5.06_my"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/17_icon_Miami_Dolphins.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 975), _c17)
except Exception:
    pass
layout["Miami_Dolphins"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 52, 68)
    canvas.paste(_c18, (1319, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/19_icon_Events_by_My_Performers.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 1520), _c19)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/20_icon_Search.png
try:
    _c20 = get_crop(20, 288, 162)
    canvas.paste(_c20, (288, 2792), _c20)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/21_icon_5.06_my.png
try:
    _c21 = get_crop(21, 59, 65)
    canvas.paste(_c21, (112, 0), _c21)
except Exception:
    pass
layout["5.06_my"] = [112, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/22_icon_Just_Announced_by_My_Performers.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1856), _c22)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/23_icon_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/24_icon_Miami_Dolphins.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1143), _c24)
except Exception:
    pass
layout["Miami_Dolphins"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/25_text_Recent_searches.png
try:
    _c25 = get_crop(25, 168, 144)
    canvas.paste(_c25, (48, 120), _c25)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_02_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-5/26_text_Suggestions.png
try:
    _c26 = get_crop(26, 331, 74)
    canvas.paste(_c26, (40, 1423), _c26)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
