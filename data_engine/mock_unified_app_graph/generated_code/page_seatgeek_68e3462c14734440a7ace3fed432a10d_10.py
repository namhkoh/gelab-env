# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_10
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13.png
# step_index: 10/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall backgrounds and structural elements for the UI page.
# Assumes `canvas` (1440x2960 RGB) and `draw` (ImageDraw) are available.

# Clear canvas to pure white background
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar (top ~72px) - light gray to match screenshot
STATUS_H = 72
draw.rectangle((0, 0, 1440, STATUS_H), fill=(243, 243, 243))
# subtle bottom border under status bar
draw.line((0, STATUS_H - 1, 1440, STATUS_H - 1), fill=(226, 226, 226), width=1)

# Hero image/background area (dark) - large black area behind the illustration
HERO_TOP = STATUS_H
HERO_BOTTOM = 520
draw.rectangle((0, HERO_TOP, 1440, HERO_BOTTOM), fill=(10, 10, 10))

# Slanted white overlay cutting into the hero (diagonal separation)
# This creates the angled white card top seen in the screenshot
draw.polygon([(0, 420), (1440, 300), (1440, HERO_BOTTOM), (0, HERO_BOTTOM)], fill=(255, 255, 255))

# Main content card (white, slightly inset) that holds title / buttons / sections
CARD_LEFT = 32
CARD_RIGHT = 1408
CARD_TOP = 300  # starts slightly overlapping the slanted area
CARD_BOTTOM = 1760
CARD_RADIUS = 20
draw.rounded_rectangle((CARD_LEFT, CARD_TOP, CARD_RIGHT, CARD_BOTTOM),
                       radius=CARD_RADIUS, fill=(255, 255, 255), outline=(235, 235, 235), width=1)

# Subtle card shadow (thin gray line to suggest elevation)
draw.line((CARD_LEFT + 6, CARD_BOTTOM + 2, CARD_RIGHT - 6, CARD_BOTTOM + 2), fill=(235, 235, 235), width=2)
draw.line((CARD_LEFT + 6, CARD_BOTTOM + 4, CARD_RIGHT - 6, CARD_BOTTOM + 4), fill=(245, 245, 245), width=1)

# Separator line between main details and location section
# Place roughly where "Get directions" area sits in the layout
SEP1_Y = 1630
draw.line((24, SEP1_Y, 1416, SEP1_Y), fill=(236, 236, 236), width=1)

# Separator above "Performers" list (another divider)
SEP2_Y = 1956
draw.line((0, SEP2_Y, 1440, SEP2_Y), fill=(236, 236, 236), width=1)

# Light background band for the performers list area to subtly distinguish it
PERF_TOP = SEP2_Y
PERF_BOTTOM = 2960
draw.rectangle((0, PERF_TOP, 1440, PERF_BOTTOM), fill=(255, 255, 255))

# Inner container for performers list (inset with subtle top padding)
PERF_INSET_LEFT = 24
PERF_INSET_RIGHT = 1416
PERF_INSET_TOP = PERF_TOP + 24
draw.rectangle((PERF_INSET_LEFT, PERF_INSET_TOP, PERF_INSET_RIGHT, PERF_BOTTOM - 24), fill=(255, 255, 255))

# Thin separators between performer rows (drawn lightly; avoid exact positions of avatars/text)
# We'll draw a few repeated separators where rows are visually expected without duplicating detected content.
row_start_y = PERF_INSET_TOP + 120
row_height = 180
for i in range(6):
    y = row_start_y + i * row_height
    if y + 1 < PERF_BOTTOM:
        draw.line((PERF_INSET_LEFT, y, PERF_INSET_RIGHT, y), fill=(245, 245, 245), width=1)

# Small subtle divider under the header/title area inside the main card
HEADER_DIV_Y = CARD_TOP + 340
draw.line((CARD_LEFT + 12, HEADER_DIV_Y, CARD_RIGHT - 12, HEADER_DIV_Y), fill=(245, 245, 245), width=1)

# Top-left back area: leave blank (do not draw the arrow icon). Instead draw a faint hit area outline (very subtle).
# We draw only a barely-visible rounded rect to convey tappable area without duplicating the icon.
BACK_AREA = (24, STATUS_H + 20, 120, STATUS_H + 92)
draw.rounded_rectangle(BACK_AREA, radius=12, outline=(250, 250, 250), width=1)

# Final subtle vignette/shading under header to separate hero and card (very light)
for i in range(6):
    alpha_y = CARD_TOP + i * 3
    shade = 250 - i  # slightly darker lines
    draw.line((CARD_LEFT + 6, alpha_y, CARD_RIGHT - 6, alpha_y), fill=(shade, shade, shade), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/00_icon_Share.png
try:
    _c0 = get_crop(0, 312, 153)
    canvas.paste(_c0, (552, 1062), _c0)
except Exception:
    pass
layout["Share"] = [552, 1062, 864, 1215]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/01_icon_Track_event.png
try:
    _c1 = get_crop(1, 444, 153)
    canvas.paste(_c1, (60, 1062), _c1)
except Exception:
    pass
layout["Track_event"] = [60, 1062, 504, 1215]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/02_icon_12_events.png
try:
    _c2 = get_crop(2, 1416, 179)
    canvas.paste(_c2, (12, 2613), _c2)
except Exception:
    pass
layout["12_events"] = [12, 2613, 1428, 2792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/03_icon_9_events.png
try:
    _c3 = get_crop(3, 1416, 168)
    canvas.paste(_c3, (12, 2792), _c3)
except Exception:
    pass
layout["9_events"] = [12, 2792, 1428, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/04_icon_35_events.png
try:
    _c4 = get_crop(4, 1416, 179)
    canvas.paste(_c4, (12, 2434), _c4)
except Exception:
    pass
layout["35_events"] = [12, 2434, 1428, 2613]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/05_icon_21events.png
try:
    _c5 = get_crop(5, 1416, 179)
    canvas.paste(_c5, (12, 2255), _c5)
except Exception:
    pass
layout["21events"] = [12, 2255, 1428, 2434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/06_icon_my.png
try:
    _c6 = get_crop(6, 57, 65)
    canvas.paste(_c6, (113, 2), _c6)
except Exception:
    pass
layout["my"] = [113, 2, 170, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/07_icon_Performers.png
try:
    _c7 = get_crop(7, 1416, 179)
    canvas.paste(_c7, (12, 2076), _c7)
except Exception:
    pass
layout["Performers"] = [12, 2076, 1428, 2255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/08_icon_Keep_the_Party_Going_A_Tribute_to.png
try:
    _c8 = get_crop(8, 444, 153)
    canvas.paste(_c8, (60, 1062), _c8)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [60, 1062, 504, 1215]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/09_icon_Keep_the_Party_Going_A_Tribute_to_Jimmy_.png
try:
    _c9 = get_crop(9, 1416, 179)
    canvas.paste(_c9, (12, 2076), _c9)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [12, 2076, 1428, 2255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/10_icon_my.png
try:
    _c10 = get_crop(10, 57, 62)
    canvas.paste(_c10, (181, 4), _c10)
except Exception:
    pass
layout["my"] = [181, 4, 238, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 51, 59)
    canvas.paste(_c11, (316, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [316, 5, 367, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 62)
    canvas.paste(_c12, (1156, 6), _c12)
except Exception:
    pass
layout["icon_12"] = [1156, 6, 1201, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 48, 63)
    canvas.paste(_c13, (1321, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [1321, 4, 1369, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 61)
    canvas.paste(_c14, (1215, 7), _c14)
except Exception:
    pass
layout["icon_14"] = [1215, 7, 1268, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 44, 61)
    canvas.paste(_c15, (1272, 6), _c15)
except Exception:
    pass
layout["icon_15"] = [1272, 6, 1316, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/16_icon_my.png
try:
    _c16 = get_crop(16, 54, 62)
    canvas.paste(_c16, (246, 4), _c16)
except Exception:
    pass
layout["my"] = [246, 4, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/17_icon_8.31.png
try:
    _c17 = get_crop(17, 104, 67)
    canvas.paste(_c17, (3, 0), _c17)
except Exception:
    pass
layout["8.31"] = [3, 0, 107, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 46, 62)
    canvas.paste(_c18, (384, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [384, 1, 430, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/19_icon_Kenny_Chesney.png
try:
    _c19 = get_crop(19, 1416, 179)
    canvas.paste(_c19, (12, 2434), _c19)
except Exception:
    pass
layout["Kenny_Chesney"] = [12, 2434, 1428, 2613]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/20_icon_Kenny_Chesney.png
try:
    _c20 = get_crop(20, 333, 58)
    canvas.paste(_c20, (245, 2466), _c20)
except Exception:
    pass
layout["Kenny_Chesney"] = [245, 2466, 578, 2524]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/21_icon_Keep_the_Party_Going_A_Tribute_to_Jimmy_.png
try:
    _c21 = get_crop(21, 1416, 179)
    canvas.paste(_c21, (12, 2255), _c21)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [12, 2255, 1428, 2434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/22_icon_Sheryl_Crow.png
try:
    _c22 = get_crop(22, 259, 57)
    canvas.paste(_c22, (247, 2287), _c22)
except Exception:
    pass
layout["Sheryl_Crow"] = [247, 2287, 506, 2344]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/23_icon_8.31.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (24, 84), _c23)
except Exception:
    pass
layout["8.31"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/24_icon_Brandi_Carlile.png
try:
    _c24 = get_crop(24, 1416, 179)
    canvas.paste(_c24, (12, 2613), _c24)
except Exception:
    pass
layout["Brandi_Carlile"] = [12, 2613, 1428, 2792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/25_text_Location.png
try:
    _c25 = get_crop(25, 212, 52)
    canvas.paste(_c25, (53, 1348), _c25)
except Exception:
    pass
layout["Location"] = [53, 1348, 265, 1400]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/26_text_Hollywood_Bowl.png
try:
    _c26 = get_crop(26, 338, 57)
    canvas.paste(_c26, (56, 1464), _c26)
except Exception:
    pass
layout["Hollywood_Bowl"] = [56, 1464, 394, 1521]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/27_text_Los_Angeles_CA_90068.png
try:
    _c27 = get_crop(27, 470, 54)
    canvas.paste(_c27, (56, 1538), _c27)
except Exception:
    pass
layout["Los_Angeles,_CA_90068"] = [56, 1538, 526, 1592]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/28_text_Get_directions.png
try:
    _c28 = get_crop(28, 1440, 113)
    canvas.paste(_c28, (0, 1637), _c28)
except Exception:
    pass
layout["Get_directions"] = [0, 1637, 1440, 1750]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/29_text_More_events_at_Hollywood_Bowl.png
try:
    _c29 = get_crop(29, 1440, 113)
    canvas.paste(_c29, (0, 1750), _c29)
except Exception:
    pass
layout["More_events_at_Hollywood_"] = [0, 1750, 1440, 1863]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/30_text_Performers.png
try:
    _c30 = get_crop(30, 255, 52)
    canvas.paste(_c30, (56, 1977), _c30)
except Exception:
    pass
layout["Performers"] = [56, 1977, 311, 2029]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/31_text_Eric_Church.png
try:
    _c31 = get_crop(31, 255, 50)
    canvas.paste(_c31, (248, 2828), _c31)
except Exception:
    pass
layout["Eric_Church"] = [248, 2828, 503, 2878]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_10_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-13/32_text_9_events.png
try:
    _c32 = get_crop(32, 177, 48)
    canvas.paste(_c32, (248, 2892), _c32)
except Exception:
    pass
layout["9_events"] = [248, 2892, 425, 2940]
