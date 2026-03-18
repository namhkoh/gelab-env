# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_11
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14.png
# step_index: 11/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar background
draw.rectangle([(0, 0), (1440, 56)], fill="#F2F2F2")

# Hero image dark background (behind illustration)
draw.rectangle([(0, 56), (1440, 420)], fill="#0B0B0B")

# Diagonal white wedge cutting into the hero (creates the slanted card top)
draw.polygon([(0, 360), (1440, 280), (1440, 420), (0, 420)], fill="#FFFFFF")

# Subtle thin divider under the hero/card area
draw.line([(32, 420), (1408, 420)], fill="#E9E9E9", width=2)

# Main content background (white) - redundant with canvas but clarifies structure
draw.rectangle([(0, 420), (1440, 2960)], fill="#FFFFFF")

# Content card behind title/buttons (rounded top to sit under diagonal)
draw.rounded_rectangle([(24, 320), (1416, 1088)], radius=12, outline=None, fill="#FFFFFF")

# Light shadow/edge under the title card (subtle separator)
draw.line([(24, 1088), (1416, 1088)], fill="#EEEEEE", width=2)

# Section separators (full-width thin lines separating sections)
separator_color = "#EAEAEA"
draw.line([(24, 1160), (1416, 1160)], fill=separator_color, width=1)   # above Location
draw.line([(24, 1680), (1416, 1680)], fill=separator_color, width=1)   # between Location and Performers
draw.line([(24, 2140), (1416, 2140)], fill=separator_color, width=1)   # after performers list
draw.line([(24, 2520), (1416, 2520)], fill=separator_color, width=1)   # above box office / CTA area

# Subtle section title gutters (light grey blocks to suggest grouping, not text)
draw.rectangle([(0, 1088), (1440, 1120)], fill="#FFFFFF")  # slight band under title card
draw.rectangle([(0, 1640), (1440, 1676)], fill="#FFFFFF")
draw.rectangle([(0, 2108), (1440, 2136)], fill="#FFFFFF")
draw.rectangle([(0, 2488), (1440, 2516)], fill="#FFFFFF")

# Thin left inset divider lines to mimic card separators (short, aligned with content margins)
left_margin = 56
right_margin = 1384
draw.line([(left_margin, 1160), (left_margin + 160, 1160)], fill=separator_color, width=1)
draw.line([(left_margin, 1680), (left_margin + 160, 1680)], fill=separator_color, width=1)
draw.line([(left_margin, 2140), (left_margin + 160, 2140)], fill=separator_color, width=1)
draw.line([(left_margin, 2520), (left_margin + 160, 2520)], fill=separator_color, width=1)

# Soft muted background band for the performers area to visually separate the list (very subtle)
draw.rectangle([(0, 1760), (1440, 2060)], fill="#FFFFFF")

# Bottom area divider (separating scroll content from sticky CTA area)
draw.line([(0, 2560), (1440, 2560)], fill="#F0F0F0", width=2)

# A faint drop-shadow line above the sticky area (not the button itself)
draw.line([(0, 2576), (1440, 2576)], fill="#F6F6F6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/00_icon_Share.png
try:
    _c0 = get_crop(0, 312, 153)
    canvas.paste(_c0, (552, 978), _c0)
except Exception:
    pass
layout["Share"] = [552, 978, 864, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/01_icon_Track_event.png
try:
    _c1 = get_crop(1, 444, 153)
    canvas.paste(_c1, (60, 978), _c1)
except Exception:
    pass
layout["Track_event"] = [60, 978, 504, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/02_icon_7_13_Wy.png
try:
    _c2 = get_crop(2, 61, 67)
    canvas.paste(_c2, (113, 0), _c2)
except Exception:
    pass
layout["7:13_Wy"] = [113, 0, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 66)
    canvas.paste(_c3, (241, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [241, 2, 305, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/04_icon_7_13_Wy.png
try:
    _c4 = get_crop(4, 53, 64)
    canvas.paste(_c4, (182, 2), _c4)
except Exception:
    pass
layout["7:13_Wy"] = [182, 2, 235, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/05_icon_Shane_Gillis.png
try:
    _c5 = get_crop(5, 444, 153)
    canvas.paste(_c5, (60, 978), _c5)
except Exception:
    pass
layout["Shane_Gillis"] = [60, 978, 504, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/06_icon_Netflix_Is_A_Joke_Fest.png
try:
    _c6 = get_crop(6, 1416, 179)
    canvas.paste(_c6, (12, 1992), _c6)
except Exception:
    pass
layout["Netflix_Is_A_Joke_Fest"] = [12, 1992, 1428, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 66)
    canvas.paste(_c7, (1154, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1154, 3, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 58, 67)
    canvas.paste(_c8, (312, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [312, 2, 370, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 49, 66)
    canvas.paste(_c9, (1321, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [1321, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/10_icon_View_tickets.png
try:
    _c10 = get_crop(10, 1440, 144)
    canvas.paste(_c10, (0, 2581), _c10)
except Exception:
    pass
layout["View_tickets"] = [0, 2581, 1440, 2725]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/11_icon_7_13_Wy.png
try:
    _c11 = get_crop(11, 110, 67)
    canvas.paste(_c11, (2, 0), _c11)
except Exception:
    pass
layout["7:13_Wy"] = [2, 0, 112, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 67)
    canvas.paste(_c12, (1213, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1213, 2, 1270, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/13_icon_Los_Angeles.png
try:
    _c13 = get_crop(13, 1440, 113)
    canvas.paste(_c13, (0, 1553), _c13)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1553, 1440, 1666]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/14_icon_Al.png
try:
    _c14 = get_crop(14, 1416, 179)
    canvas.paste(_c14, (12, 2171), _c14)
except Exception:
    pass
layout["Al"] = [12, 2171, 1428, 2350]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/15_icon_7_13_Wy.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (24, 84), _c15)
except Exception:
    pass
layout["7:13_Wy"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 49, 66)
    canvas.paste(_c16, (383, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [383, 1, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 48, 65)
    canvas.paste(_c17, (1270, 3), _c17)
except Exception:
    pass
layout["icon_17"] = [1270, 3, 1318, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/18_icon_Al.png
try:
    _c18 = get_crop(18, 1416, 179)
    canvas.paste(_c18, (12, 1992), _c18)
except Exception:
    pass
layout["Al"] = [12, 1992, 1428, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/19_icon_Netflix_Is_A_Joke_Fest.png
try:
    _c19 = get_crop(19, 1416, 179)
    canvas.paste(_c19, (12, 2171), _c19)
except Exception:
    pass
layout["Netflix_Is_A_Joke_Fest"] = [12, 2171, 1428, 2350]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/20_text_Location.png
try:
    _c20 = get_crop(20, 209, 52)
    canvas.paste(_c20, (56, 1263), _c20)
except Exception:
    pass
layout["Location"] = [56, 1263, 265, 1315]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/21_text_Performers.png
try:
    _c21 = get_crop(21, 256, 54)
    canvas.paste(_c21, (55, 1893), _c21)
except Exception:
    pass
layout["Performers"] = [55, 1893, 311, 1947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/22_text_Box_office.png
try:
    _c22 = get_crop(22, 232, 49)
    canvas.paste(_c22, (56, 2484), _c22)
except Exception:
    pass
layout["Box_office"] = [56, 2484, 288, 2533]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_11_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-14/23_clickable_More_events_at_The_Greek_Theatre_-_Los_A.png
try:
    _c23 = get_crop(23, 1440, 113)
    canvas.paste(_c23, (0, 1666), _c23)
except Exception:
    pass
layout["More_events_at_The_Greek_"] = [0, 1666, 1440, 1779]
