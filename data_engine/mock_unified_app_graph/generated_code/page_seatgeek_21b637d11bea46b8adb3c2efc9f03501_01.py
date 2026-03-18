# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_01
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4.png
# step_index: 1/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 72
status_color = "#efefef"  # very light gray
draw.rectangle([0, 0, 1440, status_h], fill=status_color)

# Subtle bottom stroke for status bar
draw.line([(0, status_h), (1440, status_h)], fill="#e0e0e0", width=1)

# Header area (location/title area)
header_top = status_h
header_h = 160
header_color = "#ffffff"
draw.rectangle([0, header_top, 1440, header_top + header_h], fill=header_color)

# Divider under header
divider_y = header_top + header_h
draw.line([(40, divider_y), (1400, divider_y)], fill="#e8e8e8", width=1)

# Hero/banner rounded card behind large promotional image
banner_x = 48
banner_y = divider_y + 24
banner_w = 1344
banner_h = 520
banner_radius = 28
banner_bg = "#0b0b0b"  # dark background for promo image area
draw.rounded_rectangle([banner_x, banner_y, banner_x + banner_w, banner_y + banner_h],
                       radius=banner_radius, fill=banner_bg)

# Slight inner border to suggest card edge (subtle)
draw.rounded_rectangle([banner_x+2, banner_y+2, banner_x + banner_w-2, banner_y + banner_h-2],
                       radius=banner_radius-2, outline="#141414", width=1)

# Section separator line below banner
sep1_y = banner_y + banner_h + 40
draw.line([(40, sep1_y), (1400, sep1_y)], fill="#f0f0f0", width=1)

# Trending events card area (white background is default, add subtle section backdrop)
trending_top = sep1_y + 24
trending_h = 380
# light panel to visually separate the list area
draw.rectangle([24, trending_top, 1416, trending_top + trending_h], fill="#ffffff")
# top divider for the list area
draw.line([(40, trending_top + 88), (1400, trending_top + 88)], fill="#ffffff", width=1)

# Draw row separators for the three trending rows (use subtle hairline)
row1_y = 1431  # approximate relative to screenshot; align separators visually
# Use calculated positions based on trending_top to avoid overlapping detected elements
r1 = trending_top + 64
r2 = r1 + 120
r3 = r2 + 120
draw.line([(40, r1), (1400, r1)], fill="#f2f2f2", width=1)
draw.line([(40, r2), (1400, r2)], fill="#f2f2f2", width=1)
draw.line([(40, r3), (1400, r3)], fill="#f2f2f2", width=1)

# Secondary divider below trending list
trending_bottom = trending_top + trending_h
draw.line([(24, trending_bottom), (1416, trending_bottom)], fill="#ededed", width=1)

# Recently viewed section background (clean white, with subtle top padding)
recent_top = trending_bottom + 28
draw.rectangle([0, recent_top, 1440, recent_top + 420], fill="#ffffff")

# Divider above recently viewed section
draw.line([(40, recent_top), (1400, recent_top)], fill="#f0f0f0", width=1)

# Recently viewed thumbnail card backgrounds (rounded) - positions match underlying detected crops
thumb_specs = [
    (48, 2382, 462, 519),   # left large thumbnail card
    (546, 2382, 462, 519),  # middle large thumbnail card
    (1044, 2382, 396, 533)  # right large thumbnail card
]
for (tx, ty, tw, th) in thumb_specs:
    # Draw white rounded card base (to represent card container)
    card_pad = 12
    rx = 22
    draw.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=rx, fill="#ffffff")
    # Inner image background (dark) to suggest image area (will be covered by pasted icons)
    img_inset = 12
    draw.rounded_rectangle([tx + img_inset, ty + img_inset,
                            tx + tw - img_inset, ty + int(th*0.62)],
                           radius=16, fill="#0f0f0f")
    # subtle card shadow line at bottom
    shadow_y = ty + th + 6
    draw.line([(tx + 6, shadow_y), (tx + tw - 6, shadow_y)], fill="#f2f2f2", width=2)

# Small horizontal divider above the very bottom navigation
nav_top = 2792
draw.line([(0, nav_top), (1440, nav_top)], fill="#eaeaea", width=1)

# Bottom navigation bar area (light background)
nav_h = 168
draw.rectangle([0, nav_top, 1440, nav_top + nav_h], fill="#ffffff")
# subtle top shadow for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e6", width=2)

# Optional subtle indicator area above nav for active tab highlight (empty, no icons drawn)
active_indicator_w = 1440 / 5
ai_x = 0
draw.rectangle([ai_x, nav_top, ai_x + active_indicator_w, nav_top + 4], fill="#fff0f0", outline=None)

# Final subtle overall vignette top/bottom edges (very light)
draw.line([(0, 0), (1440, 0)], fill="#e9e9e9", width=1)
draw.line([(0, 2959), (1440, 2959)], fill="#e9e9e9", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/00_icon_S216.png
try:
    _c0 = get_crop(0, 462, 519)
    canvas.paste(_c0, (48, 2382), _c0)
except Exception:
    pass
layout["S216+"] = [48, 2382, 510, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/01_icon_S86.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (546, 2382), _c1)
except Exception:
    pass
layout["S86+"] = [546, 2382, 1008, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/02_icon_NCAA_M_Basketball_Brooklyn.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 1667), _c2)
except Exception:
    pass
layout["NCAA_M_Basketball_Brookly"] = [0, 1667, 1309, 1903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/03_icon_St._James_Theatre.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 1431), _c3)
except Exception:
    pass
layout["St._James_Theatre"] = [0, 1431, 1309, 1667]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/04_icon_S273.png
try:
    _c4 = get_crop(4, 396, 533)
    canvas.paste(_c4, (1044, 2382), _c4)
except Exception:
    pass
layout["S273+"] = [1044, 2382, 1440, 2915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/05_icon_View_all.png
try:
    _c5 = get_crop(5, 99, 148)
    canvas.paste(_c5, (1341, 1949), _c5)
except Exception:
    pass
layout["View_all"] = [1341, 1949, 1440, 2097]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/06_icon_View_all.png
try:
    _c6 = get_crop(6, 101, 143)
    canvas.paste(_c6, (1339, 1480), _c6)
except Exception:
    pass
layout["View_all"] = [1339, 1480, 1440, 1623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/07_icon_840.png
try:
    _c7 = get_crop(7, 144, 240)
    canvas.paste(_c7, (1260, 72), _c7)
except Exception:
    pass
layout["840"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 102, 146)
    canvas.paste(_c8, (1338, 1711), _c8)
except Exception:
    pass
layout["icon_8"] = [1338, 1711, 1440, 1857]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/09_icon_Nets_at_Knicks.png
try:
    _c9 = get_crop(9, 288, 162)
    canvas.paste(_c9, (0, 2792), _c9)
except Exception:
    pass
layout["Nets_at_Knicks"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/10_icon_840.png
try:
    _c10 = get_crop(10, 97, 63)
    canvas.paste(_c10, (1217, 1), _c10)
except Exception:
    pass
layout["840"] = [1217, 1, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/11_icon_Nets_at_Knicks.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (288, 2792), _c11)
except Exception:
    pass
layout["Nets_at_Knicks"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 68)
    canvas.paste(_c12, (1155, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1155, 0, 1200, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 50, 62)
    canvas.paste(_c13, (1320, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 2, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/14_icon_GK.png
try:
    _c14 = get_crop(14, 50, 57)
    canvas.paste(_c14, (184, 4), _c14)
except Exception:
    pass
layout["GK"] = [184, 4, 234, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/15_icon_Drake_Reschedule.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (864, 2792), _c15)
except Exception:
    pass
layout["Drake_(Reschedule"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/16_icon_Andrew_Schulz.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (576, 2792), _c16)
except Exception:
    pass
layout["Andrew_Schulz"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/17_icon_View_all.png
try:
    _c17 = get_crop(17, 58, 120)
    canvas.paste(_c17, (1382, 2386), _c17)
except Exception:
    pass
layout["View_all"] = [1382, 2386, 1440, 2506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/18_icon_S2_4_D.png
try:
    _c18 = get_crop(18, 113, 127)
    canvas.paste(_c18, (1139, 1731), _c18)
except Exception:
    pass
layout["S2_(#4_D="] = [1139, 1731, 1252, 1858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/19_icon_Madison_Square_Garden.png
try:
    _c19 = get_crop(19, 1309, 234)
    canvas.paste(_c19, (0, 1903), _c19)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1903, 1309, 2137]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/20_icon_New_York_NY.png
try:
    _c20 = get_crop(20, 389, 85)
    canvas.paste(_c20, (40, 120), _c20)
except Exception:
    pass
layout["New_York,_NY"] = [40, 120, 429, 205]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/21_icon_TIcKETS.png
try:
    _c21 = get_crop(21, 1344, 840)
    canvas.paste(_c21, (48, 360), _c21)
except Exception:
    pass
layout["TIcKETS"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/22_icon_Tracking.png
try:
    _c22 = get_crop(22, 72, 72)
    canvas.paste(_c22, (906, 2406), _c22)
except Exception:
    pass
layout["Tracking"] = [906, 2406, 978, 2478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/23_text_6.37.png
try:
    _c23 = get_crop(23, 89, 45)
    canvas.paste(_c23, (20, 15), _c23)
except Exception:
    pass
layout["6.37"] = [20, 15, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/24_text_date.png
try:
    _c24 = get_crop(24, 114, 52)
    canvas.paste(_c24, (137, 208), _c24)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/25_text_Trending_events.png
try:
    _c25 = get_crop(25, 423, 79)
    canvas.paste(_c25, (38, 1303), _c25)
except Exception:
    pass
layout["Trending_events"] = [38, 1303, 461, 1382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/26_text_View_all.png
try:
    _c26 = get_crop(26, 264, 183)
    canvas.paste(_c26, (1176, 1248), _c26)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/27_text_Recently_viewed_events.png
try:
    _c27 = get_crop(27, 72, 72)
    canvas.paste(_c27, (408, 2406), _c27)
except Exception:
    pass
layout["Recently_viewed_events"] = [408, 2406, 480, 2478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/28_text_View_all.png
try:
    _c28 = get_crop(28, 264, 183)
    canvas.paste(_c28, (1176, 2199), _c28)
except Exception:
    pass
layout["View_all"] = [1176, 2199, 1440, 2382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/29_text_Nets_at_Knicks.png
try:
    _c29 = get_crop(29, 462, 519)
    canvas.paste(_c29, (48, 2382), _c29)
except Exception:
    pass
layout["Nets_at_Knicks"] = [48, 2382, 510, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/30_text_Andrew_Schulz.png
try:
    _c30 = get_crop(30, 462, 519)
    canvas.paste(_c30, (546, 2382), _c30)
except Exception:
    pass
layout["Andrew_Schulz"] = [546, 2382, 1008, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/31_text_Drake_Reschedule.png
try:
    _c31 = get_crop(31, 396, 533)
    canvas.paste(_c31, (1044, 2382), _c31)
except Exception:
    pass
layout["Drake_(Reschedule"] = [1044, 2382, 1440, 2915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_01_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-4/32_clickable_Account.png
try:
    _c32 = get_crop(32, 288, 168)
    canvas.paste(_c32, (1152, 2792), _c32)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]
