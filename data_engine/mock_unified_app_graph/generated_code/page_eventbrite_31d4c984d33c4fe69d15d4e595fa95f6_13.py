# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_13
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15.png
# step_index: 13/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background, status bar, headers, cards, separators, and nav background
# Uses provided variables: canvas (PIL Image), draw (ImageDraw)

W, H = canvas.size

# Colors
status_bar_color = (189, 189, 189)      # light gray status bar
header_bg = (255, 255, 255)             # white header area
page_bg = (255, 255, 255)               # page background (white)
divider_color = (226, 226, 226)         # thin divider lines
card_bg = (250, 250, 250)               # subtle off-white card background
card_border = (230, 230, 230)           # card border
shadow_color = (220, 220, 220)          # shadow for cards
nav_bg = (255, 255, 255)                # nav bar bg (white)
nav_top_div = (235, 235, 235)           # nav top divider

# 1) Fill overall page background (canvas starts white; do explicitly for clarity)
draw.rectangle([(0,0),(W,H)], fill=page_bg)

# 2) Status bar (top area) -- approximately 72px tall (icons are pasted on top of this)
status_h = 72
draw.rectangle([(0,0),(W,status_h)], fill=status_bar_color)

# 3) Header area (toolbar/search area) under status bar
# Detected header block occupies approx y=72..(72+191)=263
header_top = status_h
header_bottom = header_top + 191
draw.rectangle([(0, header_top), (W, header_bottom)], fill=header_bg)

# subtle bottom divider for header
draw.line([(24, header_bottom+1), (W-24, header_bottom+1)], fill=divider_color, width=2)

# 4) Filter/chips container background (leave chips themselves to be pasted)
# This is a subtle white continuation; we add a faint horizontal rule under chips area
chips_bottom = 340  # visual guidance line under chips row
draw.line([(24, chips_bottom), (W-24, chips_bottom)], fill=divider_color, width=1)

# 5) Main content card backgrounds
# Cards align with detected content x=48, width=1344 (so right = 48+1344 = 1392)
card_left = 48
card_right = 48 + 1344

# First event card container (holds image + text of first item)
first_card_top = 480
first_card_bottom = 1428
radius = 28

# Shadow behind first card
shadow_offset = 8
draw.rounded_rectangle(
    [(card_left, first_card_top + shadow_offset), (card_right, first_card_bottom + shadow_offset)],
    radius=radius, fill=shadow_color
)
# Card background and border
draw.rounded_rectangle(
    [(card_left, first_card_top), (card_right, first_card_bottom)],
    radius=radius, fill=card_bg, outline=card_border, width=1
)

# Separator between first and second card (light line to indicate spacing)
sep_y = first_card_bottom + 10
draw.line([(card_left+8, sep_y), (card_right-8, sep_y)], fill=divider_color, width=1)

# Second event card container
second_card_top = first_card_bottom + 24
second_card_bottom = min(H - 220, second_card_top + 1048 + 24)  # ensure it stays on canvas; use detected height
# Shadow behind second card
draw.rounded_rectangle(
    [(card_left, second_card_top + shadow_offset), (card_right, second_card_bottom + shadow_offset)],
    radius=radius, fill=shadow_color
)
# Card bg and border
draw.rounded_rectangle(
    [(card_left, second_card_top), (card_right, second_card_bottom)],
    radius=radius, fill=card_bg, outline=card_border, width=1
)

# 6) Additional faint horizontal separators for visual grouping (between content blocks)
draw.line([(24, second_card_bottom+18), (W-24, second_card_bottom+18)], fill=divider_color, width=1)

# 7) Bottom navigation area background and top divider
nav_top = 2804
nav_bottom = H
# top divider
draw.line([(0, nav_top), (W, nav_top)], fill=nav_top_div, width=2)
# nav background fill
draw.rectangle([(0, nav_top), (W, nav_bottom)], fill=nav_bg)

# 8) Light inner shadow at very bottom (subtle)
draw.line([(0, nav_bottom-1), (W, nav_bottom-1)], fill=(245,245,245), width=1)

# End of UI background and structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 1344, 191)
    canvas.paste(_c0, (48, 72), _c0)
except Exception:
    pass
layout["Tomorrow"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/01_icon_Music.png
try:
    _c1 = get_crop(1, 197, 111)
    canvas.paste(_c1, (875, 406), _c1)
except Exception:
    pass
layout["Music"] = [875, 406, 1072, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/02_icon_Business.png
try:
    _c2 = get_crop(2, 252, 116)
    canvas.paste(_c2, (1073, 405), _c2)
except Exception:
    pass
layout["Business"] = [1073, 405, 1325, 521]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 417, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/04_icon_Tomorrow.png
try:
    _c4 = get_crop(4, 1344, 850)
    canvas.paste(_c4, (48, 525), _c4)
except Exception:
    pass
layout["Tomorrow"] = [48, 525, 1392, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/05_icon_Business.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 697), _c5)
except Exception:
    pass
layout["Business"] = [1092, 697, 1236, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/06_icon_TICKETS.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1939), _c6)
except Exception:
    pass
layout["TICKETS"] = [1092, 1939, 1236, 2083]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/07_icon_Business.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 697), _c7)
except Exception:
    pass
layout["Business"] = [1236, 697, 1380, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/08_icon_AATO.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1939), _c8)
except Exception:
    pass
layout["AATO"] = [1236, 1939, 1380, 2083]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/09_icon_Business.png
try:
    _c9 = get_crop(9, 88, 109)
    canvas.paste(_c9, (1329, 407), _c9)
except Exception:
    pass
layout["Business"] = [1329, 407, 1417, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/10_icon_admisslon_fees_by_50.png
try:
    _c10 = get_crop(10, 1344, 1048)
    canvas.paste(_c10, (48, 1423), _c10)
except Exception:
    pass
layout["admisslon_fees_by_50%!"] = [48, 1423, 1392, 2471]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (864, 2804), _c11)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 50, 66)
    canvas.paste(_c12, (1153, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1153, 0, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 96), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 96, 66)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1308, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 61, 63)
    canvas.paste(_c15, (310, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [310, 0, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/16_icon_8.08.png
try:
    _c16 = get_crop(16, 58, 62)
    canvas.paste(_c16, (181, 1), _c16)
except Exception:
    pass
layout["8.08"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/17_icon_8.08.png
try:
    _c17 = get_crop(17, 56, 65)
    canvas.paste(_c17, (117, 0), _c17)
except Exception:
    pass
layout["8.08"] = [117, 0, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/18_icon_8.08.png
try:
    _c18 = get_crop(18, 115, 113)
    canvas.paste(_c18, (60, 114), _c18)
except Exception:
    pass
layout["8.08"] = [60, 114, 175, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/19_icon_Community_Day_1_2_Off_Admission.png
try:
    _c19 = get_crop(19, 1344, 1048)
    canvas.paste(_c19, (48, 1423), _c19)
except Exception:
    pass
layout["Community_Day_1_2_Off_Adm"] = [48, 1423, 1392, 2471]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 51, 62)
    canvas.paste(_c20, (248, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [248, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 53, 64)
    canvas.paste(_c21, (1319, 0), _c21)
except Exception:
    pass
layout["icon_21"] = [1319, 0, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/22_icon_community_events.png
try:
    _c22 = get_crop(22, 1344, 191)
    canvas.paste(_c22, (48, 72), _c22)
except Exception:
    pass
layout["community_events"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/23_icon_More.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/24_icon_Ecstatic_Dance_Full_Moon_Fusion_Cacao.png
try:
    _c24 = get_crop(24, 1344, 850)
    canvas.paste(_c24, (48, 525), _c24)
except Exception:
    pass
layout["Ecstatic_Dance_+_Full_Moo"] = [48, 525, 1392, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/25_icon_Chicago.png
try:
    _c25 = get_crop(25, 417, 144)
    canvas.paste(_c25, (0, 259), _c25)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 46, 60)
    canvas.paste(_c26, (384, 3), _c26)
except Exception:
    pass
layout["icon_26"] = [384, 3, 430, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/27_icon_Search_events.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/28_icon_community_events.png
try:
    _c28 = get_crop(28, 1344, 191)
    canvas.paste(_c28, (48, 72), _c28)
except Exception:
    pass
layout["community_events"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/29_icon_Ticket_sales_end_soon.png
try:
    _c29 = get_crop(29, 484, 86)
    canvas.paste(_c29, (91, 2115), _c29)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [91, 2115, 575, 2201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/30_icon_Home.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/31_icon_Favorites.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (576, 2804), _c31)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_13_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-15/32_text_8.08.png
try:
    _c32 = get_crop(32, 91, 43)
    canvas.paste(_c32, (20, 17), _c32)
except Exception:
    pass
layout["8.08"] = [20, 17, 111, 60]
