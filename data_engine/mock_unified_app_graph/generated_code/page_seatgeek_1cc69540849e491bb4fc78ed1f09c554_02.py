# page_id: page_seatgeek_1cc69540849e491bb4fc78ed1f09c554_02
# screenshot: 2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5.png
# step_index: 2/7
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
w, h = canvas.size
status_h = 64
status_color = (244, 245, 246)  # very light gray
draw.rectangle([(0, 0), (w, status_h)], fill=status_color)

# Subtle top hairline under status bar
draw.line([(0, status_h), (w, status_h)], fill=(224, 225, 226), width=1)

# Search bar (rounded)
search_left = 40
search_top = 74
search_right = w - 40
search_bottom = 200
search_radius = 28
search_fill = (247, 248, 249)  # off-white / light gray
search_border = (230, 231, 232)
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=search_fill,
    outline=search_border,
    width=1
)

# Thin divider under the search area
divider_y = search_bottom + 24
draw.line([(40, divider_y), (w - 40, divider_y)], fill=(234, 235, 236), width=1)

# Draw subtle separators between list rows (Recent searches list)
# Use the detected row heights (~168 px) starting from first known top around y=471
row_tops = [471, 639, 807, 975, 1143, 1311]  # boundaries between rows
for y in row_tops:
    draw.line([(40, y), (w - 40, y)], fill=(241, 241, 241), width=1)

# Heavy separator between Recent Searches and Suggestions
big_separator_y = 1320
draw.line([(24, big_separator_y), (w - 24, big_separator_y)], fill=(224, 225, 226), width=2)

# Suggestions section subtle background band (keeps contrast but doesn't draw any icons/text)
suggestions_top = big_separator_y + 24
suggestions_bottom = suggestions_top + 420
band_radius = 8
band_fill = (255, 255, 255)  # keep white but add a faint top shadow line to denote section
draw.rectangle([(24, suggestions_top), (w - 24, suggestions_bottom)], fill=band_fill)
draw.line([(24, suggestions_top), (w - 24, suggestions_top)], fill=(245, 246, 247), width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (w, h)], fill=(255, 255, 255))
draw.line([(0, nav_top), (w, nav_top)], fill=(226, 227, 228), width=1)

# Slight shadow above bottom nav to separate from content
shadow_top = nav_top - 8
for i in range(6):
    alpha = int(12 - i*2)
    if alpha <= 0:
        continue
    y = shadow_top + i
    draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

# Subtle overall page background tint (very light) to match screenshot's dominant tone
# (applies only outside main content so as not to obscure pasted elements)
edge_margin = 12
draw.rectangle([(0, status_h + 1), (w, h - (h - nav_top) - 1)], outline=None, fill=None)

# Left and right edge vertical subtle lines to frame the content area
draw.line([(24, status_h + 8), (24, nav_top - 8)], fill=(250, 250, 250), width=1)
draw.line([(w - 24, status_h + 8), (w - 24, nav_top - 8)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/00_icon_Bruno_Mars.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 639), _c0)
except Exception:
    pass
layout["Bruno_Mars"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 49, 69)
    canvas.paste(_c1, (1152, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/02_icon_Madison_Square_Garden.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 471), _c2)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 64)
    canvas.paste(_c3, (242, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [242, 2, 306, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/05_icon_L_Olympia_Olympia_Theatre.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 807), _c5)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/06_icon_Browse.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (0, 2792), _c6)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 98, 69)
    canvas.paste(_c7, (1215, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/08_icon_L_Olympia_Olympia_Theatre.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 639), _c8)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/09_icon_7.44_W.png
try:
    _c9 = get_crop(9, 90, 64)
    canvas.paste(_c9, (16, 1), _c9)
except Exception:
    pass
layout["7.44_W"] = [16, 1, 106, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/10_icon_7.44_W.png
try:
    _c10 = get_crop(10, 54, 64)
    canvas.paste(_c10, (115, 0), _c10)
except Exception:
    pass
layout["7.44_W"] = [115, 0, 169, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/11_icon_L_Olympia_Olympia_Theatre.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 975), _c11)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (576, 2792), _c12)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/13_icon_Just_Announced_by_My_Performers.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 1688), _c13)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/14_icon_7.44_W.png
try:
    _c14 = get_crop(14, 47, 63)
    canvas.paste(_c14, (186, 1), _c14)
except Exception:
    pass
layout["7.44_W"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/15_icon_Clear.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 120), _c15)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 53, 68)
    canvas.paste(_c16, (1319, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/17_icon_Account.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (1152, 2792), _c17)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/18_icon_Coldplay.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 1143), _c18)
except Exception:
    pass
layout["Coldplay"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 61, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 374, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/20_icon_7.44_W.png
try:
    _c20 = get_crop(20, 168, 144)
    canvas.paste(_c20, (48, 120), _c20)
except Exception:
    pass
layout["7.44_W"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/21_icon_Events_by_My_Performers.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 1520), _c21)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/22_icon_Coldplay.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 975), _c22)
except Exception:
    pass
layout["Coldplay"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/23_icon_Denver_Nuggets.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1143), _c23)
except Exception:
    pass
layout["Denver_Nuggets"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/25_icon_Just_Announced_by_My_Performers.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1856), _c25)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/26_icon_Performer_event_or_venue.png
try:
    _c26 = get_crop(26, 1032, 144)
    canvas.paste(_c26, (216, 120), _c26)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_02_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-5/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
