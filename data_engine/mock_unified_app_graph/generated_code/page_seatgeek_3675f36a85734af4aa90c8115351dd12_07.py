# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_07
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10.png
# step_index: 7/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the UI mockup
# Assumes provided variables: canvas (1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Base background (match dominant color - white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top ~64px) - light gray background with bottom divider
STATUS_H = 64
draw.rectangle((0, 0, 1440, STATUS_H), fill="#F5F5F5")
draw.line((0, STATUS_H, 1440, STATUS_H), fill="#E0E0E0", width=1)

# Hero image area (dark background) with a diagonal bottom edge
# Black polygon from just under status bar down to a slanted bottom
hero_top = STATUS_H
hero_right_bottom_y = 360
hero_left_bottom_y = 520
draw.polygon([
    (0, hero_top),
    (1440, hero_top),
    (1440, hero_right_bottom_y),
    (0, hero_left_bottom_y)
], fill="#0A0A0A")

# Slanted white overlay that forms the content card diagonal (keeps content area white)
# This creates the angled transition from hero image to content area
content_top_y = hero_left_bottom_y
content_right_extend = 980
draw.polygon([
    (0, content_top_y),
    (1440, hero_right_bottom_y),
    (1440, content_right_extend),
    (0, content_right_extend)
], fill="#FFFFFF")

# Subtle divider along the slanted edge
draw.line((0, content_top_y, 1440, hero_right_bottom_y), fill="#E9E9E9", width=2)
# A faint shadow line just beneath the slanted edge to imply depth
draw.line((0, content_top_y + 6, 1440, hero_right_bottom_y + 6), fill="#F2F2F2", width=4)

# Main horizontal separators between sections (full width with inner padding)
left_pad = 24
right_pad = 1440 - 24
separators = [1060, 1660, 1880, 2240]  # approximate Y positions for section dividers
for y in separators:
    draw.line((left_pad, y, right_pad, y), fill="#E9E9E9", width=1)

# Light card background/shadow for the "Performers" row area (rounded)
perf_card_top = 1920
perf_card_bottom = 2180
perf_card_left = 24
perf_card_right = 1416
draw.rounded_rectangle(
    (perf_card_left, perf_card_top, perf_card_right, perf_card_bottom),
    radius=12,
    fill="#FFFFFF",
    outline="#F0F0F0",
    width=1
)

# Subtle separator above the Box office area and a light background band
box_office_top = 2240
box_office_band_height = 520
draw.rectangle((0, box_office_top, 1440, box_office_top + box_office_band_height), fill="#FFFFFF")
draw.line((left_pad, box_office_top, right_pad, box_office_top), fill="#E9E9E9", width=1)

# Thin content grouping dividers inside the large content area (to suggest separated rows)
inside_divider_x1 = 56
inside_divider_x2 = 1384
for y in [1280, 1480, 1700]:
    draw.line((inside_divider_x1, y, inside_divider_x2, y), fill="#F3F3F3", width=1)

# Slight top shadow under the status bar to separate it from the hero area
draw.line((0, STATUS_H + 1, 1440, STATUS_H + 1), fill="#E8E8E8", width=1)

# Final subtle bottom border near the end of content to anchor the page
draw.line((0, 2920, 1440, 2920), fill="#F2F2F2", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/00_icon_Share.png
try:
    _c0 = get_crop(0, 312, 153)
    canvas.paste(_c0, (552, 978), _c0)
except Exception:
    pass
layout["Share"] = [552, 978, 864, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/01_icon_Track_event.png
try:
    _c1 = get_crop(1, 444, 153)
    canvas.paste(_c1, (60, 978), _c1)
except Exception:
    pass
layout["Track_event"] = [60, 978, 504, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/02_icon_8.11_Wy.png
try:
    _c2 = get_crop(2, 64, 67)
    canvas.paste(_c2, (110, 0), _c2)
except Exception:
    pass
layout["8.11_Wy"] = [110, 0, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 65)
    canvas.paste(_c3, (241, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [241, 3, 305, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 47, 67)
    canvas.paste(_c4, (1155, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [1155, 2, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/05_icon_8.11_Wy.png
try:
    _c5 = get_crop(5, 53, 65)
    canvas.paste(_c5, (183, 2), _c5)
except Exception:
    pass
layout["8.11_Wy"] = [183, 2, 236, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 58, 66)
    canvas.paste(_c6, (312, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [312, 3, 370, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 50, 67)
    canvas.paste(_c7, (1320, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 1, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 67)
    canvas.paste(_c8, (1215, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1215, 2, 1268, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/09_icon_View_tickets_at.png
try:
    _c9 = get_crop(9, 1440, 144)
    canvas.paste(_c9, (0, 2402), _c9)
except Exception:
    pass
layout["View_tickets_at"] = [0, 2402, 1440, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 44, 66)
    canvas.paste(_c10, (1272, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1272, 2, 1316, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/11_icon_Performers.png
try:
    _c11 = get_crop(11, 1416, 179)
    canvas.paste(_c11, (12, 1992), _c11)
except Exception:
    pass
layout["Performers"] = [12, 1992, 1428, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/12_icon_8.11_Wy.png
try:
    _c12 = get_crop(12, 109, 67)
    canvas.paste(_c12, (0, 0), _c12)
except Exception:
    pass
layout["8.11_Wy"] = [0, 0, 109, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/13_icon_8.11_Wy.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (24, 84), _c13)
except Exception:
    pass
layout["8.11_Wy"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 50, 66)
    canvas.paste(_c14, (382, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 1, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/15_text_Mkgee.png
try:
    _c15 = get_crop(15, 220, 81)
    canvas.paste(_c15, (56, 785), _c15)
except Exception:
    pass
layout["Mkgee"] = [56, 785, 276, 866]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/16_text_Location.png
try:
    _c16 = get_crop(16, 209, 52)
    canvas.paste(_c16, (56, 1263), _c16)
except Exception:
    pass
layout["Location"] = [56, 1263, 265, 1315]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/17_text_The_Fonda_Theatre.png
try:
    _c17 = get_crop(17, 411, 49)
    canvas.paste(_c17, (53, 1381), _c17)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [53, 1381, 464, 1430]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/18_text_Los_Angeles_CA_90028.png
try:
    _c18 = get_crop(18, 471, 57)
    canvas.paste(_c18, (53, 1452), _c18)
except Exception:
    pass
layout["Los_Angeles,_CA_90028"] = [53, 1452, 524, 1509]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/19_text_Get_directions.png
try:
    _c19 = get_crop(19, 1440, 113)
    canvas.paste(_c19, (0, 1553), _c19)
except Exception:
    pass
layout["Get_directions"] = [0, 1553, 1440, 1666]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/20_text_More_events_at_The_Fonda_Theatre.png
try:
    _c20 = get_crop(20, 1440, 113)
    canvas.paste(_c20, (0, 1666), _c20)
except Exception:
    pass
layout["More_events_at_The_Fonda_"] = [0, 1666, 1440, 1779]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/21_text_Performers.png
try:
    _c21 = get_crop(21, 256, 54)
    canvas.paste(_c21, (55, 1893), _c21)
except Exception:
    pass
layout["Performers"] = [55, 1893, 311, 1947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/22_text_Mkgee.png
try:
    _c22 = get_crop(22, 170, 63)
    canvas.paste(_c22, (245, 2026), _c22)
except Exception:
    pass
layout["Mkgee"] = [245, 2026, 415, 2089]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/23_text_Box_office.png
try:
    _c23 = get_crop(23, 234, 54)
    canvas.paste(_c23, (54, 2302), _c23)
except Exception:
    pass
layout["Box_office"] = [54, 2302, 288, 2356]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_07_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-10/24_clickable_Location.png
try:
    _c24 = get_crop(24, 1440, 1415)
    canvas.paste(_c24, (0, 1191), _c24)
except Exception:
    pass
layout["Location"] = [0, 1191, 1440, 2606]
