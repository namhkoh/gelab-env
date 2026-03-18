# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_05
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8.png
# step_index: 5/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint background and structural UI elements for the provided canvas.
w, h = canvas.size

# Colors (approximate to screenshot)
navy = (9, 48, 70)          # main header/navy
navy_dark = (5, 36, 54)     # status bar slightly darker
navy_deco = (18, 67, 96)    # decorative shapes
offwhite = (248, 249, 250)  # very light content card backs
divider = (229, 231, 233)   # thin separators
shadow = (240, 241, 242)    # subtle shadow line
bg_white = (255, 255, 255)  # main page white

# Fill whole canvas background (dominant color is white in the screenshot)
draw.rectangle([(0, 0), (w, h)], fill=bg_white)

# Header / banner area
banner_height = 360
draw.rectangle([(0, 0), (w, banner_height)], fill=navy)

# Status bar area (~50-90px) darker
status_h = 92
draw.rectangle([(0, 0), (w, status_h)], fill=navy_dark)

# Decorative shapes on the banner (subtle large shapes)
# Large ellipse on the right-top corner (ticket-like blob)
ellipse_bbox = (w - 520, -120, w + 120, 260)
draw.ellipse(ellipse_bbox, fill=navy_deco)

# Subtle rounded rectangle "plus" accent near center-left of banner
plus_w, plus_h = 120, 120
plus_x = int(w * 0.28)
plus_y = int(banner_height * 0.36)
draw.rounded_rectangle(
    [(plus_x, plus_y), (plus_x + plus_w, plus_y + plus_h)],
    radius=20, fill=(12, 56, 78)
)
# small cross (lighter) - using two thin rectangles to form plus
cross_color = (18, 75, 105)
cx, cy = plus_x + 24, plus_y + 30
draw.rectangle([(cx, cy + 36), (cx + 72, cy + 48)], fill=cross_color)
draw.rectangle([(cx + 36, cy), (cx + 48, cy + 72)], fill=cross_color)

# Banner bottom divider (thin)
draw.line([(0, banner_height - 1), (w, banner_height - 1)], fill=divider, width=1)

# White content area top shadow (to show separation from banner)
shadow_h = 8
for i in range(shadow_h):
    alpha = int(6 + i * 4)
    y = banner_height + i
    # simulate subtle shadow by drawing progressively lighter horizontal lines
    draw.line([(0, y), (w, y)], fill=(240 + i, 241 + i, 242 + i))

# Main content background (explicit white rectangle to ensure crisp top edge)
draw.rectangle([(0, banner_height), (w, h)], fill=bg_white)

# Draw a subtle horizontal divider under the "Buyer Guarantee" / header area.
# Based on detected text block at y ~933 with height ~126, the divider sits around y ~1060.
divider_y = 1060
draw.line([(24, divider_y), (w - 24, divider_y)], fill=divider, width=1)

# Section card backgrounds: draw soft off-white rounded rectangles behind each event row group
# Use the detected event block top positions from the provided detections:
event_tops = [1279, 1572, 1865, 2303, 2596]
card_left = 24
card_right = w - 24
card_radius = 18
card_fill = offwhite

for top in event_tops:
    bottom = top + 293  # use detected height
    # Add a small vertical inset to match UI spacing
    rect_top = top + 8
    rect_bottom = bottom - 8
    draw.rounded_rectangle(
        [(card_left, rect_top), (card_right, rect_bottom)],
        radius=card_radius, fill=card_fill
    )
    # subtle inner divider at bottom of each card area
    draw.line([(card_left + 8, rect_bottom), (card_right - 8, rect_bottom)], fill=shadow, width=1)

# Additional separators between major sections (e.g., between Upcoming Games and All Games)
# Place a separator around where "All Games" section would start (approx y ~1700)
all_games_sep_y = 1700
draw.line([(24, all_games_sep_y), (w - 24, all_games_sep_y)], fill=divider, width=1)

# Small subtle horizontal separators to visually group the first block of events (near 1280-1865)
for sep_y in [1279 + 293, 1572 + 293]:
    draw.line([(24, sep_y), (w - 24, sep_y)], fill=(245, 245, 246), width=1)

# Top-left/back area: draw a subtle circular drop shadow for where back button will be placed (but not the icon)
# Keep it faint and only a shadow so we do not duplicate the detected icon.
back_shadow_center = (84, 188)  # approximate location under the banner but overlapping
back_shadow_radius = 46
draw.ellipse(
    [
        (back_shadow_center[0] - back_shadow_radius, back_shadow_center[1] - back_shadow_radius),
        (back_shadow_center[0] + back_shadow_radius, back_shadow_center[1] + back_shadow_radius)
    ],
    fill=(255, 255, 255, 0), outline=(240, 240, 241)
)

# Final subtle full-width bottom shadow near bottom of header content area for depth
for i in range(6):
    y = banner_height + 6 + i
    alpha_shade = 242 + i
    draw.line([(0, y), (w, y)], fill=(alpha_shade, alpha_shade, alpha_shade))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/00_icon_Track_this_performer.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1104, 84), _c0)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/01_icon_Share_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 84), _c1)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/02_icon_03.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 1572), _c2)
except Exception:
    pass
layout["03"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/03_icon_23.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 2596), _c3)
except Exception:
    pass
layout["23"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/04_icon_02.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 1279), _c4)
except Exception:
    pass
layout["02"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/05_icon_22.png
try:
    _c5 = get_crop(5, 1440, 293)
    canvas.paste(_c5, (0, 2303), _c5)
except Exception:
    pass
layout["22"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/06_icon_04.png
try:
    _c6 = get_crop(6, 1440, 293)
    canvas.paste(_c6, (0, 1865), _c6)
except Exception:
    pass
layout["04"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/07_icon_8.35_Wy.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (36, 84), _c7)
except Exception:
    pass
layout["8.35_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/08_icon_New_York_Yankees.png
try:
    _c8 = get_crop(8, 202, 195)
    canvas.paste(_c8, (51, 608), _c8)
except Exception:
    pass
layout["New_York_Yankees"] = [51, 608, 253, 803]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/09_icon_Globe_Life_Field.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 1279), _c9)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 59, 64)
    canvas.paste(_c10, (243, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [243, 4, 302, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/11_icon_Globe_Life_Field.png
try:
    _c11 = get_crop(11, 1440, 293)
    canvas.paste(_c11, (0, 1572), _c11)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/12_icon_8.35_Wy.png
try:
    _c12 = get_crop(12, 56, 60)
    canvas.paste(_c12, (180, 4), _c12)
except Exception:
    pass
layout["8.35_Wy"] = [180, 4, 236, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 63, 63)
    canvas.paste(_c13, (310, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [310, 4, 373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/14_icon_Oakland_Athletics_at_New_York_Yankees.png
try:
    _c14 = get_crop(14, 1440, 293)
    canvas.paste(_c14, (0, 2303), _c14)
except Exception:
    pass
layout["Oakland_Athletics_at_New_"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/15_icon_Oakland_Athletics_at_New_York_Yankees.png
try:
    _c15 = get_crop(15, 1440, 293)
    canvas.paste(_c15, (0, 2596), _c15)
except Exception:
    pass
layout["Oakland_Athletics_at_New_"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 60, 76)
    canvas.paste(_c16, (1148, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1148, 0, 1208, 76]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/17_icon_8.35_Wy.png
try:
    _c17 = get_crop(17, 60, 64)
    canvas.paste(_c17, (115, 1), _c17)
except Exception:
    pass
layout["8.35_Wy"] = [115, 1, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 55, 72)
    canvas.paste(_c18, (382, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 2, 437, 74]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/19_icon_New_York_Yankees_at_Texas_Rangers.png
try:
    _c19 = get_crop(19, 1440, 293)
    canvas.paste(_c19, (0, 1572), _c19)
except Exception:
    pass
layout["New_York_Yankees_at_Texas"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/20_icon_New_York_Yankees_at_Texas_Rangers.png
try:
    _c20 = get_crop(20, 1440, 293)
    canvas.paste(_c20, (0, 1865), _c20)
except Exception:
    pass
layout["New_York_Yankees_at_Texas"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 86, 97)
    canvas.paste(_c21, (1306, 956), _c21)
except Exception:
    pass
layout["icon_21"] = [1306, 956, 1392, 1053]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 81, 72)
    canvas.paste(_c22, (1217, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [1217, 1, 1298, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/23_icon_S1_Hot_Dog.png
try:
    _c23 = get_crop(23, 1440, 293)
    canvas.paste(_c23, (0, 1865), _c23)
except Exception:
    pass
layout["S1_Hot_Dog"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 48, 65)
    canvas.paste(_c24, (1323, 4), _c24)
except Exception:
    pass
layout["icon_24"] = [1323, 4, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/25_icon_Oakland_Athletics_at_New_York_Yankees.png
try:
    _c25 = get_crop(25, 1440, 293)
    canvas.paste(_c25, (0, 2303), _c25)
except Exception:
    pass
layout["Oakland_Athletics_at_New_"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/26_text_New_York_Yankees.png
try:
    _c26 = get_crop(26, 526, 64)
    canvas.paste(_c26, (59, 859), _c26)
except Exception:
    pass
layout["New_York_Yankees"] = [59, 859, 585, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/27_text_Protected_by_our_Buyer_Guarantee.png
try:
    _c27 = get_crop(27, 1440, 126)
    canvas.paste(_c27, (0, 933), _c27)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/28_text_Dallas_TX.png
try:
    _c28 = get_crop(28, 271, 69)
    canvas.paste(_c28, (53, 1174), _c28)
except Exception:
    pass
layout["Dallas,_TX"] = [53, 1174, 324, 1243]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/29_text_Oakland.png
try:
    _c29 = get_crop(29, 200, 32)
    canvas.paste(_c29, (317, 2927), _c29)
except Exception:
    pass
layout["Oakland"] = [317, 2927, 517, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/30_text_Athleticc_at.png
try:
    _c30 = get_crop(30, 263, 32)
    canvas.paste(_c30, (527, 2927), _c30)
except Exception:
    pass
layout["Athleticc_at"] = [527, 2927, 790, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/31_text_New.png
try:
    _c31 = get_crop(31, 96, 29)
    canvas.paste(_c31, (800, 2930), _c31)
except Exception:
    pass
layout["New"] = [800, 2930, 896, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_05_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-8/32_text_Vork_Vankeec.png
try:
    _c32 = get_crop(32, 303, 32)
    canvas.paste(_c32, (906, 2927), _c32)
except Exception:
    pass
layout["Vork_Vankeec"] = [906, 2927, 1209, 2959]
