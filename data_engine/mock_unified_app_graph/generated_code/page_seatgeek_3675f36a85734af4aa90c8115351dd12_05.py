# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_05
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8.png
# step_index: 5/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (slightly warm off-white to match the app)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 250))

# Status bar area at the very top (~0-84px)
status_h = 84
draw.rectangle([(0, 0), (1440, status_h)], fill=(245, 245, 245))
# subtle bottom border of status bar
draw.line([(0, status_h), (1440, status_h)], fill=(225, 225, 225), width=1)

# Rounded white card that overlaps the bottom of the map (header/venue card)
card_top = 720
card_bottom = 1024
corner_radius = 28
# draw a subtle shadow band above the card to simulate elevation
shadow_h = 8
draw.rectangle([(0, card_top - shadow_h), (1440, card_top)], fill=(235, 235, 235))
# rounded white card
draw.rounded_rectangle(
    [(0, card_top), (1440, card_bottom)],
    radius=corner_radius,
    fill=(255, 255, 255),
    outline=None,
)

# thin divider under the header card (separates header from content)
divider_y = card_bottom +  -4  # slight adjust so it sits just below card
divider_y = 1016
draw.line([(48, divider_y), (1392, divider_y)], fill=(230, 230, 230), width=1)

# "Popular events" area: keep background consistent (no content drawing),
# but add a faint horizontal rule above the section to separate from header content.
popular_top_rule = 1210
draw.line([(48, popular_top_rule), (1392, popular_top_rule)], fill=(245, 245, 245), width=1)

# Separator line under the popular events cards (across full width)
# Position approximated beneath the event thumbnails
popular_bottom_div = 1878
draw.line([(24, popular_bottom_div), (1416, popular_bottom_div)], fill=(230, 230, 230), width=1)

# Seating charts section background panel (subtle off-white panel to group items)
seating_panel_top = popular_bottom_div + 12
seating_panel_bottom = 2520
panel_pad = 20
draw.rounded_rectangle(
    [(panel_pad, seating_panel_top), (1440 - panel_pad, seating_panel_bottom)],
    radius=16,
    fill=(250, 250, 250),
    outline=None,
)

# faint inner divider between seating charts and the list below
seating_bottom_div = 2508
draw.line([(48, seating_bottom_div), (1392, seating_bottom_div)], fill=(230, 230, 230), width=1)

# "All events" section background (keep it white, slightly elevated)
all_events_top = 2560
draw.rectangle([(0, all_events_top), (1440, 2960)], fill=(255, 255, 255))
# top divider for All events
draw.line([(48, all_events_top + 8), (1392, all_events_top + 8)], fill=(230, 230, 230), width=1)

# separators between list items in All events (approximate positions)
# these are subtle thin lines across content area
list_sep_positions = [2740, 2890]
for y in list_sep_positions:
    draw.line([(48, y), (1392, y)], fill=(240, 240, 240), width=1)

# Subtle left inset vertical guide to indicate content column (not a UI element)
# Keep it extremely faint so it reads as layout guide, not text/icon duplication
draw.line([(48, card_bottom), (48, 2960)], fill=(245, 245, 245), width=1)
draw.line([(1392, card_bottom), (1392, 2960)], fill=(245, 245, 245), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/00_icon_S162.png
try:
    _c0 = get_crop(0, 462, 519)
    canvas.paste(_c0, (48, 1273), _c0)
except Exception:
    pass
layout["S162+"] = [48, 1273, 510, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/01_icon_Seating_charts.png
try:
    _c1 = get_crop(1, 462, 437)
    canvas.paste(_c1, (48, 2035), _c1)
except Exception:
    pass
layout["Seating_charts"] = [48, 2035, 510, 2472]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/02_icon_Mannequin_Pussy.png
try:
    _c2 = get_crop(2, 462, 437)
    canvas.paste(_c2, (546, 2035), _c2)
except Exception:
    pass
layout["Mannequin_Pussy"] = [546, 2035, 1008, 2472]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/03_icon_S65.png
try:
    _c3 = get_crop(3, 462, 519)
    canvas.paste(_c3, (546, 1273), _c3)
except Exception:
    pass
layout["S65+"] = [546, 1273, 1008, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/04_icon_Thy_Art_Is_Murder.png
try:
    _c4 = get_crop(4, 396, 437)
    canvas.paste(_c4, (1044, 2035), _c4)
except Exception:
    pass
layout["Thy_Art_Is_Murder"] = [1044, 2035, 1440, 2472]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/05_icon_8.11_my.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 84), _c5)
except Exception:
    pass
layout["8.11_my"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/06_icon_8.11_my.png
try:
    _c6 = get_crop(6, 63, 71)
    canvas.paste(_c6, (110, 1), _c6)
except Exception:
    pass
layout["8.11_my"] = [110, 1, 173, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/07_icon_See_more_options.png
try:
    _c7 = get_crop(7, 204, 174)
    canvas.paste(_c7, (1236, 806), _c7)
except Exception:
    pass
layout["See_more_options"] = [1236, 806, 1440, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/08_icon_105.png
try:
    _c8 = get_crop(8, 1440, 704)
    canvas.paste(_c8, (0, 72), _c8)
except Exception:
    pass
layout["105"] = [0, 72, 1440, 776]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 57)
    canvas.paste(_c9, (1328, 6), _c9)
except Exception:
    pass
layout["icon_9"] = [1328, 6, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/10_icon_8.11_my.png
try:
    _c10 = get_crop(10, 107, 71)
    canvas.paste(_c10, (2, 0), _c10)
except Exception:
    pass
layout["8.11_my"] = [2, 0, 109, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/11_icon_S153.png
try:
    _c11 = get_crop(11, 396, 519)
    canvas.paste(_c11, (1044, 1273), _c11)
except Exception:
    pass
layout["S153+"] = [1044, 1273, 1440, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 59)
    canvas.paste(_c12, (1156, 9), _c12)
except Exception:
    pass
layout["icon_12"] = [1156, 9, 1201, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/13_icon_Sunset_Blvd.png
try:
    _c13 = get_crop(13, 1440, 704)
    canvas.paste(_c13, (0, 72), _c13)
except Exception:
    pass
layout["Sunset_Blvd"] = [0, 72, 1440, 776]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/14_icon_8.11_my.png
try:
    _c14 = get_crop(14, 48, 64)
    canvas.paste(_c14, (184, 4), _c14)
except Exception:
    pass
layout["8.11_my"] = [184, 4, 232, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 43, 77)
    canvas.paste(_c15, (1397, 1300), _c15)
except Exception:
    pass
layout["icon_15"] = [1397, 1300, 1440, 1377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/16_icon_Mannequin_Pussy.png
try:
    _c16 = get_crop(16, 396, 519)
    canvas.paste(_c16, (1044, 1273), _c16)
except Exception:
    pass
layout["Mannequin_Pussy"] = [1044, 1273, 1440, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 57, 64)
    canvas.paste(_c17, (245, 6), _c17)
except Exception:
    pass
layout["icon_17"] = [245, 6, 302, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/18_icon_Thy_Art_Is_Murder.png
try:
    _c18 = get_crop(18, 396, 437)
    canvas.paste(_c18, (1044, 2035), _c18)
except Exception:
    pass
layout["Thy_Art_Is_Murder"] = [1044, 2035, 1440, 2472]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/19_text_The_Fonda_Theatre.png
try:
    _c19 = get_crop(19, 546, 67)
    canvas.paste(_c19, (41, 858), _c19)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [41, 858, 587, 925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/20_text_Los_Angeles_CA.png
try:
    _c20 = get_crop(20, 371, 65)
    canvas.paste(_c20, (38, 941), _c20)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [38, 941, 409, 1006]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/21_text_Popular_events.png
try:
    _c21 = get_crop(21, 72, 72)
    canvas.paste(_c21, (408, 1297), _c21)
except Exception:
    pass
layout["Popular_events"] = [408, 1297, 480, 1369]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/22_text_Mkgee.png
try:
    _c22 = get_crop(22, 169, 63)
    canvas.paste(_c22, (42, 1626), _c22)
except Exception:
    pass
layout["Mkgee"] = [42, 1626, 211, 1689]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/23_text_d4vd.png
try:
    _c23 = get_crop(23, 119, 49)
    canvas.paste(_c23, (539, 1626), _c23)
except Exception:
    pass
layout["d4vd"] = [539, 1626, 658, 1675]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/24_text_Thu.png
try:
    _c24 = get_crop(24, 100, 51)
    canvas.paste(_c24, (42, 1695), _c24)
except Exception:
    pass
layout["Thu,"] = [42, 1695, 142, 1746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/25_text_25_9_PM.png
try:
    _c25 = get_crop(25, 183, 48)
    canvas.paste(_c25, (218, 1695), _c25)
except Exception:
    pass
layout["25,9_PM"] = [218, 1695, 401, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/26_text_Sat_Jun_29_9_PM.png
try:
    _c26 = get_crop(26, 462, 519)
    canvas.paste(_c26, (546, 1273), _c26)
except Exception:
    pass
layout["Sat,_Jun_29,9_PM"] = [546, 1273, 1008, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/27_text_Fri.png
try:
    _c27 = get_crop(27, 74, 50)
    canvas.paste(_c27, (1038, 1696), _c27)
except Exception:
    pass
layout["Fri,"] = [1038, 1696, 1112, 1746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/28_text_26_9_PM.png
try:
    _c28 = get_crop(28, 181, 48)
    canvas.paste(_c28, (1189, 1695), _c28)
except Exception:
    pass
layout["26,9_PM"] = [1189, 1695, 1370, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/29_text_Seating_charts.png
try:
    _c29 = get_crop(29, 390, 75)
    canvas.paste(_c29, (40, 1905), _c29)
except Exception:
    pass
layout["Seating_charts"] = [40, 1905, 430, 1980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/30_text_Mk.gee.png
try:
    _c30 = get_crop(30, 166, 63)
    canvas.paste(_c30, (41, 2407), _c30)
except Exception:
    pass
layout["Mk.gee"] = [41, 2407, 207, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/31_text_Mannequin_Pussy.png
try:
    _c31 = get_crop(31, 462, 437)
    canvas.paste(_c31, (546, 2035), _c31)
except Exception:
    pass
layout["Mannequin_Pussy"] = [546, 2035, 1008, 2472]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/32_text_All_events.png
try:
    _c32 = get_crop(32, 256, 54)
    canvas.paste(_c32, (46, 2590), _c32)
except Exception:
    pass
layout["All_events"] = [46, 2590, 302, 2644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/33_text_Apr_24.png
try:
    _c33 = get_crop(33, 151, 54)
    canvas.paste(_c33, (44, 2734), _c33)
except Exception:
    pass
layout["Apr_24"] = [44, 2734, 195, 2788]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/34_text_Mkgee.png
try:
    _c34 = get_crop(34, 169, 63)
    canvas.paste(_c34, (345, 2731), _c34)
except Exception:
    pass
layout["Mkgee"] = [345, 2731, 514, 2794]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/35_text_Wed.png
try:
    _c35 = get_crop(35, 103, 50)
    canvas.paste(_c35, (44, 2805), _c35)
except Exception:
    pass
layout["Wed,"] = [44, 2805, 147, 2855]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/36_text_9_PM.png
try:
    _c36 = get_crop(36, 106, 41)
    canvas.paste(_c36, (158, 2811), _c36)
except Exception:
    pass
layout["9_PM"] = [158, 2811, 264, 2852]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/37_text_S139.png
try:
    _c37 = get_crop(37, 108, 52)
    canvas.paste(_c37, (342, 2803), _c37)
except Exception:
    pass
layout["S139"] = [342, 2803, 450, 2855]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/38_text_The_Fonda_Theatre.png
try:
    _c38 = get_crop(38, 1440, 241)
    canvas.paste(_c38, (0, 2673), _c38)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [0, 2673, 1440, 2914]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/39_text_Los_Angeles_CA.png
try:
    _c39 = get_crop(39, 328, 55)
    canvas.paste(_c39, (892, 2807), _c39)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [892, 2807, 1220, 2862]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/40_clickable_Tracking.png
try:
    _c40 = get_crop(40, 144, 144)
    canvas.paste(_c40, (1260, 84), _c40)
except Exception:
    pass
layout["Tracking"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_05_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-8/41_clickable_Tracking.png
try:
    _c41 = get_crop(41, 72, 72)
    canvas.paste(_c41, (906, 1297), _c41)
except Exception:
    pass
layout["Tracking"] = [906, 1297, 978, 1369]
