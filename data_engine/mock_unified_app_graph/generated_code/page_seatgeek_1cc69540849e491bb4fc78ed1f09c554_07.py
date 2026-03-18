# page_id: page_seatgeek_1cc69540849e491bb4fc78ed1f09c554_07
# screenshot: 2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10.png
# step_index: 7/7
# task: Open SeatGeek. Search "Madison Square Garden". Select the next upcoming event. Who are the performers of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 84
draw.rectangle([(0, 0), (1440, status_h)], fill="#f6f6f6")
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill="#e6e6e6", width=1)

# Hero "image" area (dark arena-like background with subtle horizontal banding)
hero_top = status_h
hero_bottom = 520
# simple horizontal band gradient: dark bluish/charcoal bands
bands = 40
for i in range(bands):
    y0 = int(hero_top + (hero_bottom - hero_top) * (i / bands))
    y1 = int(hero_top + (hero_bottom - hero_top) * ((i + 1) / bands))
    # interpolate between two colors
    r = int(34 + (8 * i / bands))    # 34 -> 42
    g = int(30 + (20 * i / bands))   # 30 -> 50
    b = int(40 + (30 * i / bands))   # 40 -> 70
    draw.rectangle([(0, y0), (1440, y1)], fill=(r, g, b))

# subtle vignette darker edges on hero: vertical fades via thin rectangles
for i in range(16):
    alpha = int(10 + (120 * (i / 15)))
    left_x = int(0 + (i * 6))
    right_x = 1440 - left_x
    # use slightly darker rectangles to simulate edge shading
    draw.rectangle([(left_x, hero_top), (left_x + 1, hero_bottom)], fill=(10, 10, 12))
    draw.rectangle([(right_x - 1, hero_top), (right_x, hero_bottom)], fill=(10, 10, 12))

# White diagonal overlay to create the slanted lower edge of the hero image
# Points chosen to approximate the screenshot tilt
diag_poly = [
    (0, hero_bottom - 80),
    (1440, hero_bottom + 40),
    (1440, hero_bottom + 300),
    (0, hero_bottom + 160),
]
draw.polygon(diag_poly, fill="#ffffff")

# thin subtle shadow line along diagonal edge (approx by drawing a slightly darker polygon strip)
shadow_poly = [
    (0, hero_bottom - 84),
    (1440, hero_bottom + 36),
    (1440, hero_bottom + 40),
    (0, hero_bottom - 76),
]
draw.polygon(shadow_poly, fill="#efefef")

# Main content separators and structure
# Divider under the hero/title area where action buttons live
divider_y = hero_bottom + 260  # approximate zone under buttons area
draw.line([(32, divider_y), (1408, divider_y)], fill="#e9e9e9", width=1)

# Location/Info card background (subtle off-white panel with rounded corners)
loc_card_top = divider_y + 32
loc_card_bottom = loc_card_top + 520
draw.rounded_rectangle(
    [(28, loc_card_top), (1412, loc_card_bottom)],
    radius=12,
    fill="#ffffff",
    outline="#f0f0f0",
    width=1
)

# Thin separators inside location card to separate rows (do not draw any text or icons)
# "Location" header separator (top)
draw.line([(56, loc_card_top + 140), (1384, loc_card_top + 140)], fill="#f2f2f2", width=1)
# "Get directions" / "More events" separators
draw.line([(56, loc_card_top + 280), (1384, loc_card_top + 280)], fill="#f2f2f2", width=1)

# Performers section background area (distinct section with very light grey background)
perf_top = loc_card_bottom + 32
perf_bottom = perf_top + 920
draw.rectangle([(0, perf_top), (1440, perf_bottom)], fill="#ffffff")

# Section card for performers list with a subtle top border to anchor the section
draw.line([(32, perf_top + 20), (1408, perf_top + 20)], fill="#eaeaea", width=1)

# Rows for performers: draw circular placeholder backgrounds (light grey rings) but avoid drawing icons themselves.
# We'll draw faint circle background discs at the left positions for where performer avatars will be pasted.
# These are purely background discs (do not attempt to draw actual icons).
avatar_centers = [ (80, perf_top + 120), (80, perf_top + 280), (80, perf_top + 440), (80, perf_top + 600) ]
for (cx, cy) in avatar_centers:
    r_outer = 56
    r_inner = 50
    draw.ellipse([(cx - r_outer, cy - r_outer), (cx + r_outer, cy + r_outer)], fill="#fafafa", outline="#f0f0f0")
    draw.ellipse([(cx - r_inner, cy - r_inner), (cx + r_inner, cy + r_inner)], fill="#ffffff")

# Horizontal separators between performer rows
for i in range(1, 5):
    y = perf_top + 200 * i
    draw.line([(32, y), (1408, y)], fill="#f3f3f3", width=1)

# Final bottom divider to separate footer area
draw.line([(32, perf_bottom - 16), (1408, perf_bottom - 16)], fill="#eaeaea", width=1)

# Subtle overall page left/right margins shading to simulate app container
draw.rectangle([(0, perf_bottom), (1440, perf_bottom + 8)], fill="#ffffff")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/00_icon_Share.png
try:
    _c0 = get_crop(0, 312, 153)
    canvas.paste(_c0, (552, 1146), _c0)
except Exception:
    pass
layout["Share"] = [552, 1146, 864, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/01_icon_Track_event.png
try:
    _c1 = get_crop(1, 444, 153)
    canvas.paste(_c1, (60, 1146), _c1)
except Exception:
    pass
layout["Track_event"] = [60, 1146, 504, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/02_icon_Eastern_Conference_First_Round.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (24, 84), _c2)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/03_icon_7.45_Wy.png
try:
    _c3 = get_crop(3, 60, 66)
    canvas.paste(_c3, (113, 1), _c3)
except Exception:
    pass
layout["7.45_Wy"] = [113, 1, 173, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/04_icon_24_events.png
try:
    _c4 = get_crop(4, 1416, 179)
    canvas.paste(_c4, (12, 2697), _c4)
except Exception:
    pass
layout["24_events"] = [12, 2697, 1428, 2876]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/05_icon_7.45_Wy.png
try:
    _c5 = get_crop(5, 51, 64)
    canvas.paste(_c5, (183, 3), _c5)
except Exception:
    pass
layout["7.45_Wy"] = [183, 3, 234, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 61, 63)
    canvas.paste(_c6, (242, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [242, 4, 303, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/07_icon_215_events.png
try:
    _c7 = get_crop(7, 1416, 179)
    canvas.paste(_c7, (12, 2518), _c7)
except Exception:
    pass
layout["215_events"] = [12, 2518, 1428, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 65)
    canvas.paste(_c8, (1154, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [1154, 6, 1203, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/09_icon_18_events.png
try:
    _c9 = get_crop(9, 1416, 179)
    canvas.paste(_c9, (12, 2339), _c9)
except Exception:
    pass
layout["18_events"] = [12, 2339, 1428, 2518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/10_icon_NBA_Eastern_Conference_First_Round.png
try:
    _c10 = get_crop(10, 1416, 179)
    canvas.paste(_c10, (12, 2518), _c10)
except Exception:
    pass
layout["NBA_Eastern_Conference_Fi"] = [12, 2518, 1428, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 51, 67)
    canvas.paste(_c11, (1320, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1320, 2, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/12_icon_New_York_Knicks.png
try:
    _c12 = get_crop(12, 1416, 179)
    canvas.paste(_c12, (12, 2160), _c12)
except Exception:
    pass
layout["New_York_Knicks"] = [12, 2160, 1428, 2339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 57, 63)
    canvas.paste(_c13, (313, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [313, 4, 370, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/14_icon_Performers.png
try:
    _c14 = get_crop(14, 1416, 179)
    canvas.paste(_c14, (12, 2160), _c14)
except Exception:
    pass
layout["Performers"] = [12, 2160, 1428, 2339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/15_icon_Philadelphia_76ers.png
try:
    _c15 = get_crop(15, 1416, 179)
    canvas.paste(_c15, (12, 2339), _c15)
except Exception:
    pass
layout["Philadelphia_76ers"] = [12, 2339, 1428, 2518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 92, 68)
    canvas.paste(_c16, (1211, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [1211, 3, 1303, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/17_icon_7.45_Wy.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (24, 84), _c17)
except Exception:
    pass
layout["7.45_Wy"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/18_icon_Knicks_Game_2_-_Home_Game_2.png
try:
    _c18 = get_crop(18, 312, 153)
    canvas.paste(_c18, (552, 1146), _c18)
except Exception:
    pass
layout["Knicks_(Game_2_-_Home_Gam"] = [552, 1146, 864, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/19_icon_7.45_Wy.png
try:
    _c19 = get_crop(19, 111, 69)
    canvas.paste(_c19, (1, 0), _c19)
except Exception:
    pass
layout["7.45_Wy"] = [1, 0, 112, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/20_icon_New_York_Knicks.png
try:
    _c20 = get_crop(20, 1416, 179)
    canvas.paste(_c20, (12, 2160), _c20)
except Exception:
    pass
layout["New_York_Knicks"] = [12, 2160, 1428, 2339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/21_icon_Philadelphia_76ers.png
try:
    _c21 = get_crop(21, 1416, 179)
    canvas.paste(_c21, (12, 2339), _c21)
except Exception:
    pass
layout["Philadelphia_76ers"] = [12, 2339, 1428, 2518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/22_icon_NBA_Playoffs.png
try:
    _c22 = get_crop(22, 283, 55)
    canvas.paste(_c22, (244, 2552), _c22)
except Exception:
    pass
layout["NBA_Playoffs"] = [244, 2552, 527, 2607]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/23_icon_NBA_Eastern_Conference_First_Round.png
try:
    _c23 = get_crop(23, 1416, 179)
    canvas.paste(_c23, (12, 2697), _c23)
except Exception:
    pass
layout["NBA_Eastern_Conference_Fi"] = [12, 2697, 1428, 2876]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/24_icon_Madison_Square_Garden.png
try:
    _c24 = get_crop(24, 444, 153)
    canvas.paste(_c24, (60, 1146), _c24)
except Exception:
    pass
layout["Madison_Square_Garden"] = [60, 1146, 504, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/25_icon_Eastern_Conference_First_Round.png
try:
    _c25 = get_crop(25, 312, 153)
    canvas.paste(_c25, (552, 1146), _c25)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [552, 1146, 864, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 45, 64)
    canvas.paste(_c26, (383, 2), _c26)
except Exception:
    pass
layout["icon_26"] = [383, 2, 428, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/27_text_Location.png
try:
    _c27 = get_crop(27, 209, 52)
    canvas.paste(_c27, (56, 1432), _c27)
except Exception:
    pass
layout["Location"] = [56, 1432, 265, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/28_text_New_York_NY_10001.png
try:
    _c28 = get_crop(28, 390, 52)
    canvas.paste(_c28, (58, 1621), _c28)
except Exception:
    pass
layout["New_York,_NY_10001"] = [58, 1621, 448, 1673]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/29_text_Get_directions.png
try:
    _c29 = get_crop(29, 1440, 113)
    canvas.paste(_c29, (0, 1721), _c29)
except Exception:
    pass
layout["Get_directions"] = [0, 1721, 1440, 1834]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/30_text_More_events_at_Madison_Square_Garden.png
try:
    _c30 = get_crop(30, 1440, 113)
    canvas.paste(_c30, (0, 1834), _c30)
except Exception:
    pass
layout["More_events_at_Madison_Sq"] = [0, 1834, 1440, 1947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_07_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-10/31_text_Performers.png
try:
    _c31 = get_crop(31, 255, 52)
    canvas.paste(_c31, (56, 2061), _c31)
except Exception:
    pass
layout["Performers"] = [56, 2061, 311, 2113]
