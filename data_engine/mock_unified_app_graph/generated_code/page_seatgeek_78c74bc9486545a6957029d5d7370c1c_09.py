# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_09
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12.png
# step_index: 9/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 251))

# STATUS BAR (top ~50px) - darker overlay so white icons/text will show when pasted
status_bar_h = 56
draw.rectangle([(0, 0), (1440, status_bar_h)], fill=(18, 25, 35, 255))

# HERO BACKGROUND: blue textured band (behind performer image)
hero_h = 420
# Simple vertical gradient for hero area
top_blue = (27, 100, 210)
bottom_blue = (18, 72, 170)
for y in range(0, hero_h):
    t = y / max(hero_h - 1, 1)
    r = int(top_blue[0] * (1 - t) + bottom_blue[0] * t)
    g = int(top_blue[1] * (1 - t) + bottom_blue[1] * t)
    b = int(top_blue[2] * (1 - t) + bottom_blue[2] * t)
    draw.line([(0, y), (1440, y)], fill=(r, g, b))

# Add a darker irregular overlay shape to mimic the cutout / vignette from the screenshot
overlay_poly = [
    (-100, 60),
    (120, 20),
    (380, 10),
    (540, 40),
    (720, 20),
    (920, 70),
    (1140, 40),
    (1380, 80),
    (1560, 120),
    (1560, hero_h),
    (-100, hero_h)
]
draw.polygon(overlay_poly, fill=(12, 58, 122, 180))

# Very subtle speckle texture across hero (small translucent white dots)
# REMOVED: import random
rand = random.Random(1)
for _ in range(900):
    x = rand.randint(0, 1439)
    y = rand.randint(0, hero_h - 1)
    r = rand.randint(1, 2)
    draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(255, 255, 255, 20))

# Main white card under the hero (artist title area background)
card_top = hero_h - 40
card_bottom = card_top + 160
draw.rounded_rectangle([(20, card_top), (1420, card_bottom)], radius=18, fill=(255, 255, 255))
# faint divider line at bottom of this card
draw.line([(20, card_bottom), (1420, card_bottom)], fill=(230, 230, 230), width=1)

# Thin horizontal divider that separates header area from list content
divider_y = card_bottom + 40
draw.line([(0, divider_y), (1440, divider_y)], fill=(235, 235, 235), width=2)

# SECTION CARD BACKGROUNDS: draw rounded white cards behind each detected row block
# Use the provided detected vertical positions and sizes to place subtle white cards/shapes.
rows = [
    {"y": 1279, "h": 293},
    {"y": 1572, "h": 293},
    {"y": 2085, "h": 293},
    {"y": 2378, "h": 293},
    {"y": 2671, "h": 289},
]
for r in rows:
    top = r["y"] - 12  # slight padding so the rounded shape peeks out
    bottom = r["y"] + r["h"] + 12
    # draw card background across width with tiny shadow line on top
    draw.rounded_rectangle([(18, top), (1422, bottom)], radius=14, fill=(255, 255, 255))
    # subtle top emboss (thin lighter line) and bottom divider
    draw.line([(20, top), (1420, top)], fill=(250, 250, 250), width=1)
    draw.line([(20, bottom), (1420, bottom)], fill=(238, 238, 238), width=1)

# SECTION HEADERS / CONTENT AREA BACKGROUNDS
# A container for the "New York, NY" section (subtle separation)
newy_top = 1040
newy_bottom = 1700
draw.rectangle([(0, newy_top), (1440, newy_bottom)], fill=(250, 250, 251))

# A container for the "All Shows" area starting below New York section
allshows_top = 1700
allshows_bottom = 2960
draw.rectangle([(0, allshows_top), (1440, allshows_bottom)], fill=(250, 250, 251))

# Subtle separators at key section boundaries (using detected element tops to align with pasted content)
separator_positions = [933, 1279, 1572, 2085, 2378, 2671]
for y in separator_positions:
    draw.line([(24, y), (1416, y)], fill=(235, 235, 235), width=1)

# Bottom sticky area/background (to hint at footer / persistent controls)
# Use a soft translucent band at the very bottom so pasted footer icons appear distinct
footer_h = 120
draw.rectangle([(0, 2960 - footer_h), (1440, 2960)], fill=(245, 245, 246))

# Final subtle left/right padding guides (visual only, very faint) to align content blocks
draw.line([(60, divider_y + 8), (60, 2500)], fill=(250, 250, 250), width=1)
draw.line([(1380, divider_y + 8), (1380, 2500)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/00_icon_06.png
try:
    _c0 = get_crop(0, 1440, 289)
    canvas.paste(_c0, (0, 2671), _c0)
except Exception:
    pass
layout["06"] = [0, 2671, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/01_icon_Track_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1104, 84), _c1)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/02_icon_Share_this_performer.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 84), _c2)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/03_icon_Protected_by_our_Buyer_Guarantee.png
try:
    _c3 = get_crop(3, 1440, 126)
    canvas.paste(_c3, (0, 933), _c3)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/04_icon_8.29_Wy.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 84), _c4)
except Exception:
    pass
layout["8.29_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/05_icon_03.png
try:
    _c5 = get_crop(5, 1440, 293)
    canvas.paste(_c5, (0, 1279), _c5)
except Exception:
    pass
layout["03"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/06_icon_30.png
try:
    _c6 = get_crop(6, 1440, 293)
    canvas.paste(_c6, (0, 2085), _c6)
except Exception:
    pass
layout["30"] = [0, 2085, 1440, 2378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/07_icon_Madison_Square_Garden.png
try:
    _c7 = get_crop(7, 1440, 293)
    canvas.paste(_c7, (0, 1279), _c7)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/08_icon_05.png
try:
    _c8 = get_crop(8, 1440, 293)
    canvas.paste(_c8, (0, 2378), _c8)
except Exception:
    pass
layout["05"] = [0, 2378, 1440, 2671]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/09_icon_04.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 1572), _c9)
except Exception:
    pass
layout["04"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 62, 77)
    canvas.paste(_c10, (1147, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1147, 0, 1209, 77]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/11_icon_Madison_Square_Garden.png
try:
    _c11 = get_crop(11, 1440, 293)
    canvas.paste(_c11, (0, 1572), _c11)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/12_icon_8.29_Wy.png
try:
    _c12 = get_crop(12, 63, 69)
    canvas.paste(_c12, (177, 0), _c12)
except Exception:
    pass
layout["8.29_Wy"] = [177, 0, 240, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/13_icon_8.29_Wy.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (36, 84), _c13)
except Exception:
    pass
layout["8.29_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/14_icon_Inglewood_CA.png
try:
    _c14 = get_crop(14, 1440, 293)
    canvas.paste(_c14, (0, 2085), _c14)
except Exception:
    pass
layout["Inglewood,_CA"] = [0, 2085, 1440, 2378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/15_icon_Sugar_Land_TX.png
try:
    _c15 = get_crop(15, 1440, 293)
    canvas.paste(_c15, (0, 2378), _c15)
except Exception:
    pass
layout["Sugar_Land,_TX"] = [0, 2378, 1440, 2671]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 54, 65)
    canvas.paste(_c16, (315, 4), _c16)
except Exception:
    pass
layout["icon_16"] = [315, 4, 369, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/17_icon_Andrew_Schulz.png
try:
    _c17 = get_crop(17, 1440, 289)
    canvas.paste(_c17, (0, 2671), _c17)
except Exception:
    pass
layout["Andrew_Schulz"] = [0, 2671, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 94, 113)
    canvas.paste(_c18, (1300, 950), _c18)
except Exception:
    pass
layout["icon_18"] = [1300, 950, 1394, 1063]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 58, 71)
    canvas.paste(_c19, (244, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [244, 0, 302, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 103, 76)
    canvas.paste(_c20, (1217, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1217, 0, 1320, 76]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/21_text_New_York_NY.png
try:
    _c21 = get_crop(21, 350, 63)
    canvas.paste(_c21, (57, 1179), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [57, 1179, 407, 1242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/22_text_AII_Shows.png
try:
    _c22 = get_crop(22, 249, 55)
    canvas.paste(_c22, (60, 1984), _c22)
except Exception:
    pass
layout["AII_Shows"] = [60, 1984, 309, 2039]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/23_text_Andrew_Schulz.png
try:
    _c23 = get_crop(23, 1440, 289)
    canvas.paste(_c23, (0, 2671), _c23)
except Exception:
    pass
layout["Andrew_Schulz"] = [0, 2671, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_09_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-12/24_text_From_S31.png
try:
    _c24 = get_crop(24, 197, 57)
    canvas.paste(_c24, (315, 2872), _c24)
except Exception:
    pass
layout["From_S31"] = [315, 2872, 512, 2929]
