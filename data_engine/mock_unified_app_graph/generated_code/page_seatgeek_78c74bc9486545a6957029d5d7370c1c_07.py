# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_07
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10.png
# step_index: 7/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements on provided canvas and draw objects.
# Available: canvas (1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# 1) Overall background (very light warm white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 250))

# 2) Status bar area at top (~68px)
status_h = 68
draw.rectangle([(0, 0), (1440, status_h)], fill=(238, 238, 238))

# 3) Top hero/banner area (blue collage style with diagonal cut)
# Main blue banner
banner_y0 = status_h
banner_y1 = 560
banner_color = (28, 99, 196)        # primary blue
banner_dark = (16, 68, 140)         # slightly darker overlay
draw.polygon(
    [
        (0, banner_y0),
        (1440, banner_y0),
        (1440, banner_y1 - 40),
        (1100, banner_y1 - 10),
        (300, banner_y1 + 30),
        (0, banner_y1 - 10)
    ],
    fill=banner_color
)
# Decorative darker shape to imply textured cutout behind subject
draw.polygon(
    [
        (0, banner_y0 + 40),
        (1440, banner_y0 + 20),
        (1440, banner_y1 - 140),
        (900, banner_y1 - 60),
        (200, banner_y1 - 10),
        (0, banner_y1 - 20)
    ],
    fill=banner_dark
)

# 4) White header/card area that overlaps the banner with diagonal top edge
card_top = banner_y1 - 40
card_bottom = 1120
corner_radius = 20
# Use rounded rectangle for header card background
try:
    draw.rounded_rectangle(
        [(0, card_top), (1440, card_bottom)],
        radius=corner_radius,
        fill=(255, 255, 255)
    )
except Exception:
    # Fallback if rounded_rectangle not supported
    draw.rectangle([(0, card_top), (1440, card_bottom)], fill=(255, 255, 255))

# 5) Thin divider under header/content area
divider_y = card_bottom + 4
draw.line([(32, divider_y), (1408, divider_y)], fill=(230, 230, 230), width=2)

# 6) Content section blocks (subtle cards / separators)
# Location section card (white background already present) — add subtle section divider
loc_top = divider_y + 32
loc_divider = loc_top + 220
draw.line([(24, loc_divider), (1416, loc_divider)], fill=(240, 240, 240), width=1)

# More-events link area divider
more_divider = loc_divider + 140
draw.line([(24, more_divider), (1416, more_divider)], fill=(240, 240, 240), width=1)

# Performers section container (rounded subtle card)
perf_top = more_divider + 40
perf_bottom = perf_top + 240
try:
    draw.rounded_rectangle(
        [(12, perf_top), (1428, perf_bottom)],
        radius=14,
        fill=(255, 255, 255),
        outline=(242, 242, 242),
        width=1
    )
except Exception:
    draw.rectangle([(12, perf_top), (1428, perf_bottom)], fill=(255, 255, 255), outline=(242, 242, 242))

# Separator under performers
perf_sep = perf_bottom + 20
draw.line([(24, perf_sep), (1416, perf_sep)], fill=(236, 236, 236), width=1)

# Box office section area (leave large white space; draw subtle top label divider)
box_top = perf_sep + 36
box_divider = box_top + 200
draw.line([(24, box_divider), (1416, box_divider)], fill=(245, 245, 245), width=1)

# 7) Light vertical padding hints (very subtle left edge column)
# These are not content, just visual guides matching screenshot margins
left_margin_x = 56
right_margin_x = 1384
draw.line([(left_margin_x, card_top + 16), (left_margin_x, box_divider - 16)], fill=(250, 250, 250), width=1)
draw.line([(right_margin_x, card_top + 16), (right_margin_x, box_divider - 16)], fill=(250, 250, 250), width=1)

# 8) Subtle bottom area fading (do not draw the actual "View tickets" bar)
# Create a faint top shadow above where the bottom action row will be pasted later
bottom_action_top = 2360  # keep above the detected "View tickets" area at y=2402
draw.rectangle([(0, bottom_action_top - 6), (1440, bottom_action_top)], fill=(248, 248, 248))

# 9) Fine separators across the page for different logical sections
horizontal_lines = [card_bottom + 64, loc_divider + 64, perf_sep + 64, box_divider + 160]
for y in horizontal_lines:
    if y < 2400:  # avoid drawing over the bottom action bar area
        draw.line([(16, y), (1424, y)], fill=(245, 245, 245), width=1)

# End of background/structure painting.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/00_icon_Share.png
try:
    _c0 = get_crop(0, 312, 153)
    canvas.paste(_c0, (552, 978), _c0)
except Exception:
    pass
layout["Share"] = [552, 978, 864, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/01_icon_Track_event.png
try:
    _c1 = get_crop(1, 444, 153)
    canvas.paste(_c1, (60, 978), _c1)
except Exception:
    pass
layout["Track_event"] = [60, 978, 504, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/02_icon_8.28_y.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (24, 84), _c2)
except Exception:
    pass
layout["8.28_y"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 47, 68)
    canvas.paste(_c3, (1155, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [1155, 2, 1202, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 51, 68)
    canvas.paste(_c4, (1319, 1), _c4)
except Exception:
    pass
layout["icon_4"] = [1319, 1, 1370, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/05_icon_8.28_y.png
try:
    _c5 = get_crop(5, 57, 67)
    canvas.paste(_c5, (113, 1), _c5)
except Exception:
    pass
layout["8.28_y"] = [113, 1, 170, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 58)
    canvas.paste(_c6, (316, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [316, 5, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/07_icon_8.28_y.png
try:
    _c7 = get_crop(7, 58, 67)
    canvas.paste(_c7, (180, 1), _c7)
except Exception:
    pass
layout["8.28_y"] = [180, 1, 238, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/08_icon_Andrew_Schulz.png
try:
    _c8 = get_crop(8, 1416, 179)
    canvas.paste(_c8, (12, 1992), _c8)
except Exception:
    pass
layout["Andrew_Schulz"] = [12, 1992, 1428, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/09_icon_View_tickets.png
try:
    _c9 = get_crop(9, 1440, 144)
    canvas.paste(_c9, (0, 2402), _c9)
except Exception:
    pass
layout["View_tickets"] = [0, 2402, 1440, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 65)
    canvas.paste(_c10, (246, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [246, 2, 300, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/11_icon_Share.png
try:
    _c11 = get_crop(11, 444, 153)
    canvas.paste(_c11, (60, 978), _c11)
except Exception:
    pass
layout["Share"] = [60, 978, 504, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/12_icon_Performers.png
try:
    _c12 = get_crop(12, 1416, 179)
    canvas.paste(_c12, (12, 1992), _c12)
except Exception:
    pass
layout["Performers"] = [12, 1992, 1428, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 64, 70)
    canvas.paste(_c13, (1213, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1213, 0, 1277, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 46, 64)
    canvas.paste(_c14, (383, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [383, 0, 429, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 47, 66)
    canvas.paste(_c15, (1270, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [1270, 3, 1317, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/16_icon_More_events_at_Madison_Square_Garden.png
try:
    _c16 = get_crop(16, 1440, 1415)
    canvas.paste(_c16, (0, 1191), _c16)
except Exception:
    pass
layout["More_events_at_Madison_Sq"] = [0, 1191, 1440, 2606]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/17_text_Location.png
try:
    _c17 = get_crop(17, 209, 52)
    canvas.paste(_c17, (56, 1263), _c17)
except Exception:
    pass
layout["Location"] = [56, 1263, 265, 1315]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/18_text_Get_directions.png
try:
    _c18 = get_crop(18, 1440, 113)
    canvas.paste(_c18, (0, 1553), _c18)
except Exception:
    pass
layout["Get_directions"] = [0, 1553, 1440, 1666]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/19_text_More_events_at_Madison_Square_Garden.png
try:
    _c19 = get_crop(19, 1440, 113)
    canvas.paste(_c19, (0, 1666), _c19)
except Exception:
    pass
layout["More_events_at_Madison_Sq"] = [0, 1666, 1440, 1779]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/20_text_Performers.png
try:
    _c20 = get_crop(20, 256, 54)
    canvas.paste(_c20, (55, 1893), _c20)
except Exception:
    pass
layout["Performers"] = [55, 1893, 311, 1947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_07_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-10/21_text_Box_office.png
try:
    _c21 = get_crop(21, 234, 54)
    canvas.paste(_c21, (54, 2302), _c21)
except Exception:
    pass
layout["Box_office"] = [54, 2302, 288, 2356]
