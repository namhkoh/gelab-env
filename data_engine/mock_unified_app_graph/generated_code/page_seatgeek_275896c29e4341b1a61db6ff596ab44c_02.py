# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_02
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5.png
# step_index: 2/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (slightly off-white to match the app background)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 250))

# Status bar area (top)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(245, 245, 245))
# subtle bottom border for status bar
draw.line([(0, status_h), (1440, status_h)], fill=(225, 225, 225), width=2)

# Search bar / header background (rounded)
search_box = (48, 96, 1392, 216)
draw.rounded_rectangle(search_box, radius=36, fill=(245, 245, 245), outline=(230, 230, 230), width=1)

# Divider under search area
divider_y = 232
draw.line([(32, divider_y), (1408, divider_y)], fill=(230, 230, 230), width=2)

# Section separators / subtle dividers (full width)
separators = [
    480,  # near top of recent searches area
    660,  # after a few recent items
    840,  # between stacked rows
    1020,
    1200,
    1400, # above suggestions area
    1600,
    1760
]
for y in separators:
    draw.line([(24, y), (1416, y)], fill=(240, 240, 240), width=1)

# Suggestion section card background (light rounded panel)
suggest_card = (32, 1360, 1408, 1920)
draw.rounded_rectangle(suggest_card, radius=14, fill=(250, 250, 250), outline=(245, 245, 245))

# Subtle horizontal rule separating Recent Searches and Suggestions
draw.line([(24, 720), (1416, 720)], fill=(230, 230, 230), width=1)

# Bottom navigation bar background (dock)
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
# Top shadow / divider for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill=(220, 220, 220), width=2)

# Small subtle vertical padding lines to suggest content columns (light)
draw.line([(48, 280), (48, 2600)], fill=(255, 255, 255, 0))
draw.line([(1392, 280), (1392, 2600)], fill=(255, 255, 255, 0))

# Slight left gutter background (just a faint panel to convey structure)
draw.rectangle([(24, 232), (1416, 2800)], outline=(245, 245, 245), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/00_icon_Bruno_Mars.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 639), _c0)
except Exception:
    pass
layout["Bruno_Mars"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 49, 69)
    canvas.paste(_c1, (1152, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/02_icon_Madison_Square_Garden.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 471), _c2)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 64)
    canvas.paste(_c3, (242, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [242, 2, 306, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/05_icon_L_Olympia_Olympia_Theatre.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 807), _c5)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/06_icon_Browse.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (0, 2792), _c6)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 98, 69)
    canvas.paste(_c7, (1215, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/08_icon_L_Olympia_Olympia_Theatre.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 639), _c8)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/09_icon_L_Olympia_Olympia_Theatre.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 975), _c9)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (576, 2792), _c10)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/11_icon_Just_Announced_by_My_Performers.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 1688), _c11)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/13_icon_Coldplay.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 1143), _c13)
except Exception:
    pass
layout["Coldplay"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/14_icon_7.48_Wy.png
try:
    _c14 = get_crop(14, 47, 63)
    canvas.paste(_c14, (186, 1), _c14)
except Exception:
    pass
layout["7.48_Wy"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 68)
    canvas.paste(_c15, (1319, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/17_icon_7.48_Wy.png
try:
    _c17 = get_crop(17, 168, 144)
    canvas.paste(_c17, (48, 120), _c17)
except Exception:
    pass
layout["7.48_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 64)
    canvas.paste(_c18, (313, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [313, 2, 375, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/19_icon_Coldplay.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 975), _c19)
except Exception:
    pass
layout["Coldplay"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/20_icon_Events_by_My_Performers.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 1520), _c20)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/21_icon_7.48_Wy.png
try:
    _c21 = get_crop(21, 57, 65)
    canvas.paste(_c21, (113, 0), _c21)
except Exception:
    pass
layout["7.48_Wy"] = [113, 0, 170, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/22_icon_Denver_Nuggets.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["Denver_Nuggets"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/25_icon_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 168, 144)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_02_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-5/27_text_Suggestions.png
try:
    _c27 = get_crop(27, 331, 74)
    canvas.paste(_c27, (40, 1423), _c27)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
