# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_05
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8.png
# step_index: 5/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for a 1440x2960 canvas using PIL ImageDraw (variables provided).
# Available: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill overall background (white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top subtle gray)
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill="#F2F2F2")

# Subtle bottom border of status bar
draw.line([(0, status_h), (1440, status_h)], fill="#E0E0E0", width=1)

# Toolbar area below status bar (keeps white but separated)
toolbar_top = status_h
toolbar_bottom = 220
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill="#FFFFFF")

# Search input / toolbar search pill (rounded)
search_left = 80
search_top = 120
search_right = 1360
search_bottom = 196
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=36,
    fill="#FAFAFA",
    outline="#E6E6E6",
    width=2
)

# Divider under toolbar / search area
divider_y = toolbar_bottom + 4
draw.line([(40, divider_y), (1400, divider_y)], fill="#E9E9E9", width=1)

# Main suggestions block background (subtle card-like area behind suggestion rows)
# This sits below the divider and groups the first two location suggestions
card_left = 24
card_top = divider_y + 24
card_right = 1416
card_bottom = 620
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=12,
    fill="#FBFBFB",
    outline="#F1F1F1",
    width=1
)

# Separator lines between suggestion rows inside the card
# Approx positions for two rows (visual separators)
row1_bottom = 420
row2_bottom = 560
sep_x1 = card_left + 24
sep_x2 = card_right - 24
draw.line([(sep_x1, row1_bottom), (sep_x2, row1_bottom)], fill="#E8E8E8", width=1)
draw.line([(sep_x1, row2_bottom), (sep_x2, row2_bottom)], fill="#E8E8E8", width=1)

# Thin divider below suggestions card before "Recent locations" area
below_card_divider_y = card_bottom + 36
draw.line([(40, below_card_divider_y), (1400, below_card_divider_y)], fill="#EFEFEF", width=1)

# "Recent locations" header area spacing (no text drawn, only subtle underline)
recent_header_top = below_card_divider_y + 24
recent_header_bottom = recent_header_top + 96
# subtle highlight block behind header for visual separation (very light)
draw.rectangle([(0, recent_header_top - 8), (1440, recent_header_bottom)], fill="#FFFFFF")

# Separator lines for recent location items (two items)
recent_item1_y = recent_header_bottom + 80
recent_item2_y = recent_item1_y + 168
draw.line([(40, recent_item1_y), (1400, recent_item1_y)], fill="#F0F0F0", width=1)
draw.line([(40, recent_item2_y), (1400, recent_item2_y)], fill="#F0F0F0", width=1)

# Large whitespace/content area remains white (no drawing to avoid overlapping pasted elements)

# Subtle right-edge and left-edge margins (visual guides only, extremely light)
draw.line([(24, 0), (24, 2960)], fill="#FFFFFF", width=1)
draw.line([(1416, 0), (1416, 2960)], fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 53, 69)
    canvas.paste(_c0, (1150, 1), _c0)
except Exception:
    pass
layout["icon_0"] = [1150, 1, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/01_icon_7_10_my.png
try:
    _c1 = get_crop(1, 168, 144)
    canvas.paste(_c1, (0, 122), _c1)
except Exception:
    pass
layout["7:10_my"] = [0, 122, 168, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 101, 69)
    canvas.paste(_c2, (1212, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1212, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 66, 66)
    canvas.paste(_c3, (241, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [241, 2, 307, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/04_icon_Los_Angeles_CA.png
try:
    _c4 = get_crop(4, 125, 140)
    canvas.paste(_c4, (44, 1027), _c4)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [44, 1027, 169, 1167]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/05_icon_Los_Angeles.png
try:
    _c5 = get_crop(5, 95, 121)
    canvas.paste(_c5, (224, 135), _c5)
except Exception:
    pass
layout["Los_Angeles"] = [224, 135, 319, 256]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 53, 67)
    canvas.paste(_c6, (1319, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1319, 1, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 63, 65)
    canvas.paste(_c7, (312, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [312, 2, 375, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/08_icon_7_10_my.png
try:
    _c8 = get_crop(8, 50, 65)
    canvas.paste(_c8, (184, 1), _c8)
except Exception:
    pass
layout["7:10_my"] = [184, 1, 234, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/09_icon_Clear.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 122), _c9)
except Exception:
    pass
layout["Clear"] = [1248, 122, 1392, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/10_icon_Los_Angeles_CA.png
try:
    _c10 = get_crop(10, 113, 124)
    canvas.paste(_c10, (47, 346), _c10)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [47, 346, 160, 470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/11_icon_Recent_locations.png
try:
    _c11 = get_crop(11, 121, 132)
    canvas.paste(_c11, (44, 508), _c11)
except Exception:
    pass
layout["Recent_locations"] = [44, 508, 165, 640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 936, 144)
    canvas.paste(_c12, (312, 122), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [312, 122, 1248, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/13_text_7_10_my.png
try:
    _c13 = get_crop(13, 153, 52)
    canvas.paste(_c13, (19, 9), _c13)
except Exception:
    pass
layout["7:10_my"] = [19, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/14_text_Los_Angeles_CA.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 316), _c14)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 316, 1440, 484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/15_text_West_Los_Angeles_CA.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 484), _c15)
except Exception:
    pass
layout["West_Los_Angeles,_CA"] = [0, 484, 1440, 652]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/16_text_Recent_locations.png
try:
    _c16 = get_crop(16, 441, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Recent_locations"] = [44, 742, 485, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/17_text_New_York_NY.png
try:
    _c17 = get_crop(17, 299, 56)
    canvas.paste(_c17, (213, 894), _c17)
except Exception:
    pass
layout["New_York,_NY"] = [213, 894, 512, 950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/18_text_Los_Angeles_CA.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 1005), _c18)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1005, 1440, 1173]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_05_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-8/19_clickable_New_York_NY.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 837), _c19)
except Exception:
    pass
layout["New_York,_NY"] = [0, 837, 1440, 1005]
