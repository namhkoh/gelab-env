# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_06
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9.png
# step_index: 6/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 250))

# Status bar area (dark translucent look)
STATUS_H = 88
draw.rectangle([(0, 0), (1440, STATUS_H)], fill=(25, 25, 25))

# Soft gold banner behind the artwork (top hero background)
BANNER_TOP = STATUS_H
BANNER_BOTTOM = 940
draw.rectangle([(0, BANNER_TOP), (1440, BANNER_BOTTOM)], fill=(247, 237, 220))

# Decorative gold jagged edges at left and right to echo the screenshot sides
def jagged_edge(x_start, x_end, y_top, y_bottom, steps, color):
    step_h = (y_bottom - y_top) / steps
    pts = []
    x = x_start
    y = y_top
    pts.append((x, y))
    for i in range(steps + 1):
        y = y_top + i * step_h
        offset = (i % 2) * (x_end - x_start) * 0.9
        if x_start < x_end:
            pts.append((x + offset, y))
        else:
            pts.append((x - offset, y))
    pts.append((x, y_bottom))
    return pts

left_poly = jagged_edge(0, 260, BANNER_TOP, BANNER_BOTTOM, 18, None)
draw.polygon(left_poly, fill=(223, 179, 84))

right_poly = jagged_edge(1440, 1180, BANNER_TOP, BANNER_BOTTOM, 18, None)
draw.polygon(right_poly, fill=(223, 179, 84))

# Subtle radial-ish highlight behind the main artwork area (centered)
cx, cy = 720, 360
for i, col in enumerate([(255,255,255), (250,246,239), (245,237,217), (238,224,187)]):
    r = 520 - i * 120
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=col)

# White content card under the banner (title area)
TITLE_TOP = BANNER_BOTTOM - 20
TITLE_BOTTOM = 1120
draw.rounded_rectangle([(28, TITLE_TOP), (1440 - 28, TITLE_BOTTOM)], radius=18, fill=(255,255,255))

# Drop shadow under title card
shadow_y = TITLE_BOTTOM
for i, alpha in enumerate([18, 12, 8]):
    draw.rectangle([(28, shadow_y + i),(1440-28, shadow_y + i + 1)], fill=(220,220,220))

# Thin divider under title card
draw.line([(28 + 8, TITLE_BOTTOM + 6), (1440 - 28 - 8, TITLE_BOTTOM + 6)], fill=(230,230,230), width=1)

# Main list container (group holding show rows)
LIST_TOP = TITLE_BOTTOM + 28
LIST_LEFT = 0
LIST_RIGHT = 1440
LIST_BOTTOM = 2920
draw.rectangle([(LIST_LEFT, LIST_TOP), (LIST_RIGHT, LIST_BOTTOM)], fill=(250,250,250))

# Card strip behind the series of shows (subtle white cards for grouping)
GROUP_PAD_L = 36
GROUP_PAD_R = 36
group_y = LIST_TOP + 24
group_h = 1560
draw.rounded_rectangle([(GROUP_PAD_L, group_y), (1440 - GROUP_PAD_R, group_y + group_h)], radius=12, fill=(255,255,255))

# Horizontal separators between list rows (aligned near detected row positions)
seps = [1279, 1572, 1865, 2158, 2440, 2596]  # using given y positions and a couple extras
for y in seps:
    # draw subtle divider across the inner card area
    draw.line([(GROUP_PAD_L + 18, y), (1440 - GROUP_PAD_R - 18, y)], fill=(238,238,238), width=1)

# Light shadow for the group card bottom
for i in range(6):
    alpha = 10 - i
    y = group_y + group_h + i
    draw.rectangle([(GROUP_PAD_L + 2, y), (1440 - GROUP_PAD_R - 2, y + 1)], fill=(230,230,230))

# Section header area above "All Shows" (separator + tiny background)
ALL_SHOWS_DIV_Y = 2460
draw.line([(36, ALL_SHOWS_DIV_Y - 18), (1440 - 36, ALL_SHOWS_DIV_Y - 18)], fill=(245,245,245), width=1)
draw.rectangle([(36, ALL_SHOWS_DIV_Y - 8), (1440 - 36, ALL_SHOWS_DIV_Y + 64)], fill=(250,250,250))

# Small floating card style background for date-pill placeholders on the left (do not draw the pill content)
# We'll draw rounded transparent placeholders behind where the date chips appear, but keep them very subtle.
date_positions = [1279, 1572, 1865, 2158, 2596]
for y in date_positions:
    top = y + 12
    left = 36
    right = left + 150
    bottom = top + 170
    draw.rounded_rectangle([(left, top), (right, bottom)], radius=18, fill=(249,249,249))

# Final subtle vertical divider on the very left to frame content
draw.rectangle([(0, TITLE_BOTTOM + 2), (6, LIST_BOTTOM)], fill=(245,245,245))

# End of background/structure drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/00_icon_Eugene_0_Neill_Theatre.png
try:
    _c0 = get_crop(0, 1440, 293)
    canvas.paste(_c0, (0, 1279), _c0)
except Exception:
    pass
layout["Eugene_0'Neill_Theatre"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/01_icon_Eugene_0_Neill_Theatre.png
try:
    _c1 = get_crop(1, 1440, 293)
    canvas.paste(_c1, (0, 1572), _c1)
except Exception:
    pass
layout["Eugene_0'Neill_Theatre"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/02_icon_24.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 1572), _c2)
except Exception:
    pass
layout["24"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/03_icon_23.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 1279), _c3)
except Exception:
    pass
layout["23"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/04_icon_25.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 1865), _c4)
except Exception:
    pass
layout["25"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/05_icon_23.png
try:
    _c5 = get_crop(5, 1440, 293)
    canvas.paste(_c5, (0, 2596), _c5)
except Exception:
    pass
layout["23"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/06_icon_MORM.png
try:
    _c6 = get_crop(6, 1440, 126)
    canvas.paste(_c6, (0, 933), _c6)
except Exception:
    pass
layout["MORM"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/07_icon_Eugene_0_Neill_Theatre.png
try:
    _c7 = get_crop(7, 1440, 293)
    canvas.paste(_c7, (0, 1865), _c7)
except Exception:
    pass
layout["Eugene_0'Neill_Theatre"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/08_icon_6.51Wv.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 84), _c8)
except Exception:
    pass
layout["6.51Wv"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/09_icon_26.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 2158), _c9)
except Exception:
    pass
layout["26"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/10_icon_New_York_NY.png
try:
    _c10 = get_crop(10, 1440, 293)
    canvas.paste(_c10, (0, 2596), _c10)
except Exception:
    pass
layout["New_York,_NY"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/11_icon_14.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1260, 84), _c11)
except Exception:
    pass
layout["14"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 103, 116)
    canvas.paste(_c12, (1297, 945), _c12)
except Exception:
    pass
layout["icon_12"] = [1297, 945, 1400, 1061]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/13_icon_14.png
try:
    _c13 = get_crop(13, 105, 88)
    canvas.paste(_c13, (1116, 0), _c13)
except Exception:
    pass
layout["14"] = [1116, 0, 1221, 88]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/14_icon_14.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1104, 84), _c14)
except Exception:
    pass
layout["14"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/15_icon_Eugene_0_Neill_Theatre.png
try:
    _c15 = get_crop(15, 1440, 293)
    canvas.paste(_c15, (0, 2158), _c15)
except Exception:
    pass
layout["Eugene_0'Neill_Theatre"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 56, 63)
    canvas.paste(_c16, (1319, 968), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 968, 1375, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/17_text_6.51Wv.png
try:
    _c17 = get_crop(17, 149, 45)
    canvas.paste(_c17, (22, 13), _c17)
except Exception:
    pass
layout["6.51Wv"] = [22, 13, 171, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/18_text_14.png
try:
    _c18 = get_crop(18, 27, 39)
    canvas.paste(_c18, (1330, 16), _c18)
except Exception:
    pass
layout["14"] = [1330, 16, 1357, 55]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/19_text_New_York_NY.png
try:
    _c19 = get_crop(19, 352, 65)
    canvas.paste(_c19, (55, 1177), _c19)
except Exception:
    pass
layout["New_York,_NY"] = [55, 1177, 407, 1242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/20_text_AII_Shows.png
try:
    _c20 = get_crop(20, 249, 55)
    canvas.paste(_c20, (60, 2495), _c20)
except Exception:
    pass
layout["AII_Shows"] = [60, 2495, 309, 2550]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_06_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-9/21_text_The_Rook_Of_Mormon.png
try:
    _c21 = get_crop(21, 1440, 293)
    canvas.paste(_c21, (0, 2596), _c21)
except Exception:
    pass
layout["The_Rook_Of_Mormon"] = [0, 2596, 1440, 2889]
