# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_05
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8.png
# step_index: 5/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top hero gradient (warm orange) and overall page structure backgrounds
hero_height = 640
w, h = canvas.size

# Gradient from darker orange to lighter orange
top_col = (180, 60, 10)   # deep orange
bottom_col = (240, 120, 40)  # lighter orange
for y in range(hero_height):
    t = y / max(hero_height - 1, 1)
    r = int(top_col[0] * (1 - t) + bottom_col[0] * t)
    g = int(top_col[1] * (1 - t) + bottom_col[1] * t)
    b = int(top_col[2] * (1 - t) + bottom_col[2] * t)
    draw.line([(0, y), (w, y)], fill=(r, g, b))

# Status bar overlay (darker band at very top)
status_h = 110
status_color = (140, 40, 8)  # slightly darker orange for status bar area
draw.rectangle([0, 0, w, status_h], fill=status_color)

# White rounded content panel that overlaps the hero image
panel_top = 560
panel_radius = 36
draw.rounded_rectangle([0, panel_top, w, h], radius=panel_radius, fill=(255, 255, 255))

# Subtle shadow line above the white panel (soft separator)
shadow_top = panel_top - 12
shadow_bottom = panel_top
draw.rectangle([0, shadow_top, w, shadow_bottom], fill=(240, 240, 240))

# Thin divider line at very top of content area (subtle)
draw.line([(24, panel_top + 8), (w - 24, panel_top + 8)], fill=(230, 230, 230), width=1)

# Section separators between list rows (light rules across content)
sep_color = (242, 242, 242)
x_left = 32
x_right = w - 32
# start a bit below the top of the white panel and repeat at regular intervals
y = panel_top + 220
while y < h - 160:
    draw.line([(x_left, y), (x_right, y)], fill=sep_color, width=1)
    y += 220

# Faint grouping background bands to subtly separate larger sections
band_color = (250, 250, 250)
# First band under title area
draw.rectangle([0, panel_top + 24, w, panel_top + 140], fill=band_color)
# Second band further down (for "All Shows" grouping area)
draw.rectangle([0, panel_top + 920, w, panel_top + 1040], fill=band_color)

# Bottom edge safe area subtle gradient (to give depth)
for i in range(40):
    alpha = int(245 - (i * 2))  # pseudo darkening
    y0 = h - 40 + i
    if y0 < h:
        color = (alpha, alpha, alpha)
        draw.line([(0, y0), (w, y0)], fill=color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/00_icon_Track_this_performer.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1104, 84), _c0)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/01_icon_Share_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 84), _c1)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/02_icon_The_Lion_King.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 1279), _c2)
except Exception:
    pass
layout["The_Lion_King"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/03_icon_7.40.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 84), _c3)
except Exception:
    pass
layout["7.40"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/04_icon_7.00_PM.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 1572), _c4)
except Exception:
    pass
layout["7.00_PM"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/05_icon_7_00_PM.png
try:
    _c5 = get_crop(5, 1440, 293)
    canvas.paste(_c5, (0, 1865), _c5)
except Exception:
    pass
layout["7:00_PM"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/06_icon_20.png
try:
    _c6 = get_crop(6, 1440, 293)
    canvas.paste(_c6, (0, 1572), _c6)
except Exception:
    pass
layout["20"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/07_icon_Protected_by_our_Buyer_Guarantee.png
try:
    _c7 = get_crop(7, 1440, 126)
    canvas.paste(_c7, (0, 933), _c7)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 70)
    canvas.paste(_c8, (1152, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1152, 2, 1204, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/09_icon_20.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 1279), _c9)
except Exception:
    pass
layout["20"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/10_icon_New_York_NY.png
try:
    _c10 = get_crop(10, 1440, 293)
    canvas.paste(_c10, (0, 2596), _c10)
except Exception:
    pass
layout["New_York,_NY"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/11_icon_21.png
try:
    _c11 = get_crop(11, 1440, 293)
    canvas.paste(_c11, (0, 1865), _c11)
except Exception:
    pass
layout["21"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 93, 71)
    canvas.paste(_c12, (1212, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1212, 1, 1305, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/13_icon_8.00_PM.png
try:
    _c13 = get_crop(13, 1440, 293)
    canvas.paste(_c13, (0, 2158), _c13)
except Exception:
    pass
layout["8.00_PM"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/14_icon_SEEK.png
try:
    _c14 = get_crop(14, 61, 72)
    canvas.paste(_c14, (178, 1), _c14)
except Exception:
    pass
layout["SEEK"] = [178, 1, 239, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/15_icon_SEEK.png
try:
    _c15 = get_crop(15, 62, 70)
    canvas.paste(_c15, (242, 3), _c15)
except Exception:
    pass
layout["SEEK"] = [242, 3, 304, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 52, 72)
    canvas.paste(_c16, (1320, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1320, 1, 1372, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/17_icon_The_Lion_King.png
try:
    _c17 = get_crop(17, 1440, 293)
    canvas.paste(_c17, (0, 1865), _c17)
except Exception:
    pass
layout["The_Lion_King"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/18_icon_7.40.png
try:
    _c18 = get_crop(18, 160, 70)
    canvas.paste(_c18, (11, 1), _c18)
except Exception:
    pass
layout["7.40"] = [11, 1, 171, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/19_icon_The_Lion_King.png
try:
    _c19 = get_crop(19, 1440, 293)
    canvas.paste(_c19, (0, 1572), _c19)
except Exception:
    pass
layout["The_Lion_King"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 92, 107)
    canvas.paste(_c20, (1302, 949), _c20)
except Exception:
    pass
layout["icon_20"] = [1302, 949, 1394, 1056]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/21_icon_The_Lion_King.png
try:
    _c21 = get_crop(21, 1440, 293)
    canvas.paste(_c21, (0, 2158), _c21)
except Exception:
    pass
layout["The_Lion_King"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/22_icon_20.png
try:
    _c22 = get_crop(22, 1440, 293)
    canvas.paste(_c22, (0, 2596), _c22)
except Exception:
    pass
layout["20"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/23_text_New_York_NY.png
try:
    _c23 = get_crop(23, 350, 63)
    canvas.paste(_c23, (57, 1179), _c23)
except Exception:
    pass
layout["New_York,_NY"] = [57, 1179, 407, 1242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/24_text_AII_Shows.png
try:
    _c24 = get_crop(24, 249, 55)
    canvas.paste(_c24, (60, 2495), _c24)
except Exception:
    pass
layout["AII_Shows"] = [60, 2495, 309, 2550]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/25_text_The.png
try:
    _c25 = get_crop(25, 89, 29)
    canvas.paste(_c25, (317, 2930), _c25)
except Exception:
    pass
layout["The"] = [317, 2930, 406, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/26_text_ion_Kina.png
try:
    _c26 = get_crop(26, 194, 32)
    canvas.paste(_c26, (437, 2927), _c26)
except Exception:
    pass
layout["ion_Kina"] = [437, 2927, 631, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_05_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-8/27_text_New_Vork.png
try:
    _c27 = get_crop(27, 210, 29)
    canvas.paste(_c27, (673, 2930), _c27)
except Exception:
    pass
layout["New_Vork"] = [673, 2930, 883, 2959]
