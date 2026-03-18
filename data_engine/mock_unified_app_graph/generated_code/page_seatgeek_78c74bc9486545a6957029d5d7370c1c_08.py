# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_08
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11.png
# step_index: 8/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top hero gradient (blue) - tall header background
hero_top = 0
hero_bottom = 640
start_color = (10, 58, 127)   # deep blue
end_color = (57, 126, 222)    # lighter blue

for y in range(hero_top, hero_bottom):
    t = (y - hero_top) / max(1, hero_bottom - hero_top - 1)
    r = int(start_color[0] * (1 - t) + end_color[0] * t)
    g = int(start_color[1] * (1 - t) + end_color[1] * t)
    b = int(start_color[2] * (1 - t) + end_color[2] * t)
    draw.line([(0, y), (1440, y)], fill=(r, g, b))

# Status bar area at the very top (~88px) - slightly darker overlay to host white icons later
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill=(6, 38, 86))

# Soft top vignette (subtle darker band under status bar)
for i in range(20):
    alpha = int(10 - i * 0.4)
    if alpha <= 0:
        break
    y = status_h + i
    # simulate subtle darker band by drawing slightly darker lines
    draw.line([(0, y), (1440, y)], fill=(6 - i, 38 - i, 86 - i))

# White rounded card that bridges hero and content (title area background)
card_top = 560
card_bottom = 980
card_radius = 20
# Using rounded_rectangle if available
try:
    draw.rounded_rectangle([(24, card_top), (1440 - 24, card_bottom)], radius=card_radius, fill=(255, 255, 255))
except Exception:
    # fallback: plain rectangle if rounded not available
    draw.rectangle([(24, card_top), (1440 - 24, card_bottom)], fill=(255, 255, 255))

# Subtle shadow below the white card
shadow_top = card_bottom
for i in range(12):
    y = shadow_top + i
    alpha = int(30 - i * 2)
    if alpha <= 0:
        break
    grey = 230 - i
    draw.line([(24, y), (1440 - 24, y)], fill=(grey, grey, grey))

# Thin divider under the buyer-guarantee area (approx where content sections separate)
divider_y_positions = [930, 1279, 1572, 1876, 2085, 2378, 2671, 2872]
for y in divider_y_positions:
    draw.line([(24, y), (1440 - 24, y)], fill=(235, 235, 235), width=2)

# Section group backgrounds (subtle off-white panels behind groups of rows)
# New York section panel
try:
    draw.rounded_rectangle([(24, 1060), (1440 - 24, 1500)], radius=12, fill=(255, 255, 255))
except Exception:
    draw.rectangle([(24, 1060), (1440 - 24, 1500)], fill=(255, 255, 255))

# All Shows panel
try:
    draw.rounded_rectangle([(24, 1860), (1440 - 24, 2680)], radius=12, fill=(255, 255, 255))
except Exception:
    draw.rectangle([(24, 1860), (1440 - 24, 2680)], fill=(255, 255, 255))

# Light left margin guideline (visual structure only)
draw.line([(24, card_top), (24, 2960)], fill=(248, 248, 248), width=1)
draw.line([(1440 - 24, card_top), (1440 - 24, 2960)], fill=(248, 248, 248), width=1)

# Subtle horizontal spacing lines to separate list rows within the All Shows area
row_start_x = 36
row_end_x = 1440 - 36
row_ys = [1960, 2140, 2320, 2500, 2680]
for y in row_ys:
    draw.line([(row_start_x, y), (row_end_x, y)], fill=(245, 245, 245), width=1)

# Decorative faint rounded background for the date badge column (very subtle)
badge_col_x = 36
badge_col_w = 150
try:
    draw.rounded_rectangle([(badge_col_x, 1200), (badge_col_x + badge_col_w, 2680)], radius=12, fill=(250, 250, 250))
except Exception:
    draw.rectangle([(badge_col_x, 1200), (badge_col_x + badge_col_w, 2680)], fill=(250, 250, 250))

# Bottom large footer safety area background (slightly off-white)
footer_top = 2860
draw.rectangle([(0, footer_top), (1440, 2960)], fill=(249, 249, 249))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/00_icon_Track_this_performer.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1104, 84), _c0)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/01_icon_Share_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 84), _c1)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/02_icon_06.png
try:
    _c2 = get_crop(2, 1440, 289)
    canvas.paste(_c2, (0, 2671), _c2)
except Exception:
    pass
layout["06"] = [0, 2671, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/03_icon_Protected_by_our_Buyer_Guarantee.png
try:
    _c3 = get_crop(3, 1440, 126)
    canvas.paste(_c3, (0, 933), _c3)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/04_icon_8.28_Wy.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 84), _c4)
except Exception:
    pass
layout["8.28_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/05_icon_30.png
try:
    _c5 = get_crop(5, 1440, 293)
    canvas.paste(_c5, (0, 2085), _c5)
except Exception:
    pass
layout["30"] = [0, 2085, 1440, 2378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/06_icon_03.png
try:
    _c6 = get_crop(6, 1440, 293)
    canvas.paste(_c6, (0, 1279), _c6)
except Exception:
    pass
layout["03"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/07_icon_Madison_Square_Garden.png
try:
    _c7 = get_crop(7, 1440, 293)
    canvas.paste(_c7, (0, 1279), _c7)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/08_icon_04.png
try:
    _c8 = get_crop(8, 1440, 293)
    canvas.paste(_c8, (0, 1572), _c8)
except Exception:
    pass
layout["04"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/09_icon_05.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 2378), _c9)
except Exception:
    pass
layout["05"] = [0, 2378, 1440, 2671]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 62, 76)
    canvas.paste(_c10, (1147, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1147, 0, 1209, 76]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/11_icon_8.28_Wy.png
try:
    _c11 = get_crop(11, 63, 69)
    canvas.paste(_c11, (177, 0), _c11)
except Exception:
    pass
layout["8.28_Wy"] = [177, 0, 240, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/12_icon_Madison_Square_Garden.png
try:
    _c12 = get_crop(12, 1440, 293)
    canvas.paste(_c12, (0, 1572), _c12)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/13_icon_8.28_Wy.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (36, 84), _c13)
except Exception:
    pass
layout["8.28_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 65)
    canvas.paste(_c14, (315, 4), _c14)
except Exception:
    pass
layout["icon_14"] = [315, 4, 369, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/15_icon_Inglewood_CA.png
try:
    _c15 = get_crop(15, 1440, 293)
    canvas.paste(_c15, (0, 2085), _c15)
except Exception:
    pass
layout["Inglewood,_CA"] = [0, 2085, 1440, 2378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/16_icon_Sugar_Land_TX.png
try:
    _c16 = get_crop(16, 1440, 293)
    canvas.paste(_c16, (0, 2378), _c16)
except Exception:
    pass
layout["Sugar_Land,_TX"] = [0, 2378, 1440, 2671]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/17_icon_Andrew_Schulz.png
try:
    _c17 = get_crop(17, 1440, 289)
    canvas.paste(_c17, (0, 2671), _c17)
except Exception:
    pass
layout["Andrew_Schulz"] = [0, 2671, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 94, 113)
    canvas.paste(_c18, (1300, 950), _c18)
except Exception:
    pass
layout["icon_18"] = [1300, 950, 1394, 1063]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 57, 70)
    canvas.paste(_c19, (244, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [244, 0, 301, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 100, 74)
    canvas.paste(_c20, (1215, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1215, 0, 1315, 74]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/21_text_New_York_NY.png
try:
    _c21 = get_crop(21, 350, 63)
    canvas.paste(_c21, (57, 1179), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [57, 1179, 407, 1242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/22_text_AII_Shows.png
try:
    _c22 = get_crop(22, 249, 55)
    canvas.paste(_c22, (60, 1984), _c22)
except Exception:
    pass
layout["AII_Shows"] = [60, 1984, 309, 2039]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/23_text_Andrew_Schulz.png
try:
    _c23 = get_crop(23, 1440, 289)
    canvas.paste(_c23, (0, 2671), _c23)
except Exception:
    pass
layout["Andrew_Schulz"] = [0, 2671, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_08_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-11/24_text_From_S31.png
try:
    _c24 = get_crop(24, 197, 57)
    canvas.paste(_c24, (315, 2872), _c24)
except Exception:
    pass
layout["From_S31"] = [315, 2872, 512, 2929]
