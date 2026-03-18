# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_07
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9.png
# step_index: 7/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile UI page
# Uses provided variables: canvas (1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background fill (soft off-white / pale grey)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 251, 252))

# Status bar area (top ~80px) - darker band behind status icons
STATUS_H = 80
draw.rectangle([(0, 0), (1440, STATUS_H)], fill=(110, 115, 120))

# Header / toolbar background area beneath status bar
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 240
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_BOTTOM)], fill=(255, 255, 255))

# Subtle divider under header
draw.line([(40, HEADER_BOTTOM), (1400, HEADER_BOTTOM)], fill=(226, 227, 230), width=2)

# Light horizontal rule to separate filter chips / section from header
FILTER_DIV_Y = 520
draw.line([(40, FILTER_DIV_Y), (1400, FILTER_DIV_Y)], fill=(241, 242, 244), width=1)

# Card 1 container (rounded white card with subtle offset shadow)
card1_x0, card1_y0 = 36, 620
card1_x1, card1_y1 = 1404, 1500  # container behind first event block
shadow_offset = 10
# Shadow (offset, light gray)
draw.rounded_rectangle(
    [(card1_x0 + shadow_offset, card1_y0 + shadow_offset),
     (card1_x1 + shadow_offset, card1_y1 + shadow_offset)],
    radius=28,
    fill=(236, 238, 241)
)
# Card fill (white) and thin border
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=28,
    fill=(255, 255, 255),
    outline=(235, 237, 240),
    width=2
)

# Divider line between image area and text area inside card1 (visual structure only)
# Place it a bit below the expected image area so text overlays will be pasted above it
draw.line([(card1_x0 + 36, card1_y0 + 420), (card1_x1 - 36, card1_y0 + 420)], fill=(245, 246, 248), width=1)

# Small rounded tag background placeholder area under image (keeps spacing without drawing text)
# (This is just a soft rounded rectangle to match card structure, not an icon/text)
tag_x0, tag_y0 = card1_x0 + 36, card1_y0 + 440
tag_x1, tag_y1 = tag_x0 + 220, tag_y0 + 52
draw.rounded_rectangle([(tag_x0, tag_y0), (tag_x1, tag_y1)], radius=18, fill=(246, 243, 253))

# Card 2 container (rounded white card with subtle offset shadow)
card2_x0, card2_y0 = 36, 1760
card2_x1, card2_y1 = 1404, 2580
# Shadow
draw.rounded_rectangle(
    [(card2_x0 + shadow_offset, card2_y0 + shadow_offset),
     (card2_x1 + shadow_offset, card2_y1 + shadow_offset)],
    radius=28,
    fill=(236, 238, 241)
)
# Card fill and border
draw.rounded_rectangle(
    [(card2_x0, card2_y0), (card2_x1, card2_y1)],
    radius=28,
    fill=(255, 255, 255),
    outline=(235, 237, 240),
    width=2
)

# Divider inside card2 for spacing of image area vs metadata (visual only)
draw.line([(card2_x0 + 36, card2_y0 + 430), (card2_x1 - 36, card2_y0 + 430)], fill=(245, 246, 248), width=1)

# Light gray section separators between stacked content
sep_positions = [card1_y1 + 28, card2_y1 + 28]
for y in sep_positions:
    draw.line([(40, y), (1400, y)], fill=(245, 246, 247), width=1)

# Bottom navigation bar background and top divider
NAV_TOP = 2804
NAV_BOTTOM = 2960
# Divider line above nav bar
draw.line([(0, NAV_TOP), (1440, NAV_TOP)], fill=(225, 226, 229), width=2)
# Nav bar fill
draw.rectangle([(0, NAV_TOP), (1440, NAV_BOTTOM)], fill=(255, 255, 255))

# Add subtle indicator areas for nav (no icons/text, just structural circles for spacing)
nav_center_y = NAV_TOP + (NAV_BOTTOM - NAV_TOP) // 2
nav_x_positions = [72, 360, 720, 1080, 1368]
for x in nav_x_positions:
    draw.ellipse([(x - 36, nav_center_y - 36), (x + 36, nav_center_y + 36)], outline=(245, 246, 248), width=1)

# Left panel accent (thin vertical guide line near left content edge)
draw.line([(40, HEADER_BOTTOM + 12), (40, NAV_TOP - 12)], fill=(248, 249, 250), width=1)

# Right panel accent (thin vertical guide line near right content edge)
draw.line([(1400, HEADER_BOTTOM + 12), (1400, NAV_TOP - 12)], fill=(248, 249, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/04_icon_sow.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2355), _c4)
except Exception:
    pass
layout["sow"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/05_icon_sow.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2355), _c5)
except Exception:
    pass
layout["sow"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/06_icon_Foo.png
try:
    _c6 = get_crop(6, 146, 110)
    canvas.paste(_c6, (1283, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1429, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/07_icon_Q.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["~Q)"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/08_icon_Q.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["~Q)"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 57, 62)
    canvas.paste(_c9, (246, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [246, 1, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/10_icon_9.42.png
try:
    _c10 = get_crop(10, 123, 112)
    canvas.paste(_c10, (57, 116), _c10)
except Exception:
    pass
layout["9.42"] = [57, 116, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 1344, 191)
    canvas.paste(_c11, (48, 72), _c11)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 64)
    canvas.paste(_c12, (1152, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1152, 0, 1205, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 61, 63)
    canvas.paste(_c13, (311, 1), _c13)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 93, 61)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1305, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/15_icon_9.42.png
try:
    _c15 = get_crop(15, 55, 62)
    canvas.paste(_c15, (182, 0), _c15)
except Exception:
    pass
layout["9.42"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/16_icon_Los_Angeles.png
try:
    _c16 = get_crop(16, 492, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 59, 59)
    canvas.paste(_c17, (1317, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1317, 0, 1376, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/18_icon_9.42.png
try:
    _c18 = get_crop(18, 57, 64)
    canvas.paste(_c18, (114, 0), _c18)
except Exception:
    pass
layout["9.42"] = [114, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 51, 60)
    canvas.paste(_c19, (383, 3), _c19)
except Exception:
    pass
layout["Search_forae"] = [383, 3, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/20_icon_LAFW_CELEBRITY_RUNWAY_SHOW.png
try:
    _c20 = get_crop(20, 1344, 1115)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["LAFW_CELEBRITY_RUNWAY_SHO"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/21_icon_Best_Comedv_Club_Near_Me_Theater.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["Best_Comedv_Club_Near_Me_"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 242, 62)
    canvas.paste(_c22, (85, 1686), _c22)
except Exception:
    pass
layout["Promoted"] = [85, 1686, 327, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/23_icon_anel.png
try:
    _c23 = get_crop(23, 1344, 977)
    canvas.paste(_c23, (48, 1839), _c23)
except Exception:
    pass
layout["anel%"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/24_icon_Tickets.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/25_icon_3422_West_Pico_Boulevard_Los_Angeles_CA_.png
try:
    _c25 = get_crop(25, 1344, 1115)
    canvas.paste(_c25, (48, 676), _c25)
except Exception:
    pass
layout["3422_West_Pico_Boulevard;"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/26_icon_Best_Comedv_Club_Near_Me_Theater.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Best_Comedv_Club_Near_Me_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/27_text_9.42.png
try:
    _c27 = get_crop(27, 91, 41)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["9.42"] = [20, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/28_text_10_000_events.png
try:
    _c28 = get_crop(28, 359, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/29_clickable_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_07_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-9/30_clickable_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
