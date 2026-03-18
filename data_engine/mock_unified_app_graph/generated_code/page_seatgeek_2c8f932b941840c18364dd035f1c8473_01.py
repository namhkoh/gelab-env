# page_id: page_seatgeek_2c8f932b941840c18364dd035f1c8473_01
# screenshot: 2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4.png
# step_index: 1/8
# task: Open SeatGeek. Search "Beatles Love". Select the soonest upcoming event. Choose 2 tickets and continue. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (dominant off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")

# Status bar area (top ~80px) - light grey background where system icons will be pasted
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#E9E9E9")
# subtle bottom stroke for status bar
draw.line([(0, status_h-1), (1440, status_h-1)], fill="#DADADA", width=1)

# Header / toolbar area (under status bar, light / white)
header_top = status_h
header_h = 120
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill="#FFFFFF")
# header bottom divider
draw.line([(24, header_top + header_h), (1440-24, header_top + header_h)], fill="#E7E7E7", width=1)

# Large horizontal divider under the big banner (banner ends ~y=1200; leave spacing then divider)
# Place a soft divider between banner and the "Just for you" section (around y=1280)
divider_y = 1280
draw.line([(24, divider_y), (1440-24, divider_y)], fill="#EDEDED", width=1)

# Light subtle section band behind list content (trending area)
# Keep padding so icons/text areas (detected) remain visually on top
trending_top = 1560
trending_bottom = 2640
band_radius = 8
draw.rounded_rectangle(
    [(12, trending_top), (1440-12, trending_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline=None
)
# Add a faint top separator for the trending band
draw.line([(24, trending_top), (1440-24, trending_top)], fill="#E9E9E9", width=1)

# Row separators for trending list items (soft lines)
sep_x0 = 24
sep_x1 = 1440 - 24
seps = [1880, 2100, 2320, 2540]
for y in seps:
    draw.line([(sep_x0, y), (sep_x1, y)], fill="#F0F0F0", width=1)

# Subtle left edge shadow to visually separate content from edge
shadow = [(0, header_top + 10), (8, trending_bottom)]
draw.rectangle(shadow, fill="#FBFBFB")

# Bottom navigation bar area (draw background and top divider)
nav_top = 2720
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(24, nav_top), (1440-24, nav_top)], fill="#E6E6E6", width=1)

# Small visual accent: a faint vertical divider near the right edge for UI balance
draw.line([(1440-72, header_top + 12), (1440-72, trending_bottom)], fill="#FAFAFA", width=1)

# Decorative subtle horizontal guides (very light) to aid pasted elements alignment (non-intrusive)
guide_color = "#FFFFFF"
guide_ys = [320, 520, 760, 1040, 1400, 1680]
for y in guide_ys:
    draw.line([(48, y), (1440-48, y)], fill=guide_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/00_icon_S94.png
try:
    _c0 = get_crop(0, 462, 519)
    canvas.paste(_c0, (48, 1431), _c0)
except Exception:
    pass
layout["S94+"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/01_icon_Knicks.png
try:
    _c1 = get_crop(1, 1344, 840)
    canvas.paste(_c1, (48, 360), _c1)
except Exception:
    pass
layout["Knicks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/02_icon_August_Wilson_Theatre.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 2183), _c2)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/03_icon_S116.png
try:
    _c3 = get_crop(3, 462, 519)
    canvas.paste(_c3, (546, 1431), _c3)
except Exception:
    pass
layout["S116+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/04_icon_Yankee_Stadium.png
try:
    _c4 = get_crop(4, 1309, 236)
    canvas.paste(_c4, (0, 2419), _c4)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 100, 152)
    canvas.paste(_c5, (1340, 2464), _c5)
except Exception:
    pass
layout["icon_5"] = [1340, 2464, 1440, 2616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/06_icon_View_all.png
try:
    _c6 = get_crop(6, 99, 151)
    canvas.paste(_c6, (1341, 2227), _c6)
except Exception:
    pass
layout["View_all"] = [1341, 2227, 1440, 2378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/07_icon_New_York_NY.png
try:
    _c7 = get_crop(7, 64, 58)
    canvas.paste(_c7, (242, 5), _c7)
except Exception:
    pass
layout["New_York,_NY"] = [242, 5, 306, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/08_icon_888.png
try:
    _c8 = get_crop(8, 144, 240)
    canvas.paste(_c8, (1260, 72), _c8)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/09_icon_5.06_my.png
try:
    _c9 = get_crop(9, 55, 57)
    canvas.paste(_c9, (114, 5), _c9)
except Exception:
    pass
layout["5.06_my"] = [114, 5, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/10_icon_888.png
try:
    _c10 = get_crop(10, 99, 65)
    canvas.paste(_c10, (1214, 0), _c10)
except Exception:
    pass
layout["888"] = [1214, 0, 1313, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/11_icon_5.06_my.png
try:
    _c11 = get_crop(11, 50, 56)
    canvas.paste(_c11, (184, 6), _c11)
except Exception:
    pass
layout["5.06_my"] = [184, 6, 234, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/12_icon_Apr.png
try:
    _c12 = get_crop(12, 264, 183)
    canvas.paste(_c12, (1176, 2000), _c12)
except Exception:
    pass
layout["Apr"] = [1176, 2000, 1440, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/13_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c13 = get_crop(13, 288, 168)
    canvas.paste(_c13, (864, 2792), _c13)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 64)
    canvas.paste(_c14, (1319, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1319, 1, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 47, 66)
    canvas.paste(_c15, (1154, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1154, 0, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/16_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (576, 2792), _c16)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/17_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (288, 2792), _c17)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 101, 119)
    canvas.paste(_c18, (1339, 2697), _c18)
except Exception:
    pass
layout["icon_18"] = [1339, 2697, 1440, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/19_icon_Browse.png
try:
    _c19 = get_crop(19, 288, 162)
    canvas.paste(_c19, (0, 2792), _c19)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 54, 59)
    canvas.paste(_c21, (316, 5), _c21)
except Exception:
    pass
layout["icon_21"] = [316, 5, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/22_icon_S116.png
try:
    _c22 = get_crop(22, 462, 519)
    canvas.paste(_c22, (546, 1431), _c22)
except Exception:
    pass
layout["S116+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 116, 128)
    canvas.paste(_c23, (1138, 2483), _c23)
except Exception:
    pass
layout["icon_23"] = [1138, 2483, 1254, 2611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/24_icon_New_York_NY.png
try:
    _c24 = get_crop(24, 390, 87)
    canvas.paste(_c24, (40, 119), _c24)
except Exception:
    pass
layout["New_York,_NY"] = [40, 119, 430, 206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/25_text_date.png
try:
    _c25 = get_crop(25, 114, 52)
    canvas.paste(_c25, (137, 208), _c25)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/26_text_Just_for_you.png
try:
    _c26 = get_crop(26, 306, 66)
    canvas.paste(_c26, (38, 1310), _c26)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/27_text_View_all.png
try:
    _c27 = get_crop(27, 264, 183)
    canvas.paste(_c27, (1176, 1248), _c27)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/28_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c28 = get_crop(28, 288, 168)
    canvas.paste(_c28, (576, 2792), _c28)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/29_clickable_Tracking.png
try:
    _c29 = get_crop(29, 72, 72)
    canvas.paste(_c29, (408, 1455), _c29)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_01_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-4/30_clickable_Tracking.png
try:
    _c30 = get_crop(30, 72, 72)
    canvas.paste(_c30, (906, 1455), _c30)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
