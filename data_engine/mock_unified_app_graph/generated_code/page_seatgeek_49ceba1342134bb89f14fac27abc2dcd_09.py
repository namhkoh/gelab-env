# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_09
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12.png
# step_index: 9/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 60
draw.rectangle([(0, 0), (1440, status_h)], fill=(244, 244, 244))

# Thin divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(230, 230, 230), width=1)

# Header / toolbar background area (behind search area and toolbar)
header_top = status_h
header_bottom = 240
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Subtle bottom divider for header
draw.line([(48, header_bottom), (1392, header_bottom)], fill=(235, 235, 235), width=2)

# Main content subtle background band (keeps page feeling bright / off-white)
content_band_top = header_bottom + 8
content_band_bottom = 2200
draw.rectangle([(0, content_band_top), (1440, content_band_bottom)], fill=(255, 255, 255))

# Separator lines between major sections (subtle grey)
separators = [
    1311,  # divider after Recent searches list
    1512,  # between Suggestions header and content group
    1696,  # between Events by My Performers and Just Announced groups
    1864,  # lower section divider
]
for y in separators:
    draw.line([(48, y), (1392, y)], fill=(238, 238, 238), width=2)

# Soft rounded card behind the "Recent searches" area (no content drawn)
recent_card_top = 200
recent_card_bottom = 1311
draw.rounded_rectangle(
    [(36, recent_card_top), (1404, recent_card_bottom)],
    radius=18,
    fill=(255, 255, 255),
    outline=(245, 245, 245),
    width=1
)

# Subtle separators between list items (light lines spaced to match item heights)
item_y = 471
item_height = 168
for i in range(1, 5):  # draw a few internal separators for visual structure
    y = item_y + i * item_height
    draw.line([(36 + 96, y), (1404 - 48, y)], fill=(247, 247, 247), width=1)

# Suggestions area background (very slight tint to separate from lists)
suggestions_top = 1311 + 16
suggestions_bottom = 2100
draw.rectangle([(0, suggestions_top), (1440, suggestions_bottom)], fill=(255, 255, 255))

# Bottom navigation bar background and top border/shadow
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
# top border line for nav
draw.line([(0, nav_top), (1440, nav_top)], fill=(230, 230, 230), width=2)
# soft shadow under nav (very subtle)
for i, a in enumerate([6, 4, 2]):
    draw.line([(0, nav_top + i + 1), (1440, nav_top + i + 1)], fill=(245, 245, 245), width=1)

# Small visual guide dots at left/right edges to indicate safe area (very subtle, non-content)
draw.ellipse([(16, header_top + 8), (26, header_top + 18)], fill=(250, 250, 250))
draw.ellipse([(1414, header_top + 8), (1424, header_top + 18)], fill=(250, 250, 250))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 70)
    canvas.paste(_c0, (1153, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/01_icon_Music_Hall.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 471), _c1)
except Exception:
    pass
layout["Music_Hall"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/02_icon_8.35_my.png
try:
    _c2 = get_crop(2, 168, 144)
    canvas.paste(_c2, (48, 120), _c2)
except Exception:
    pass
layout["8.35_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/03_icon_Tracking.png
try:
    _c3 = get_crop(3, 288, 168)
    canvas.paste(_c3, (864, 2792), _c3)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/04_icon_Suggestions.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 1143), _c4)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/05_icon_Browse.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (0, 2792), _c5)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/06_icon_Shin_Lim.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 975), _c6)
except Exception:
    pass
layout["Shin_Lim"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 63, 64)
    canvas.paste(_c7, (242, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [242, 2, 305, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 96, 69)
    canvas.paste(_c8, (1216, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1216, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/10_icon_Music_Hall.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 639), _c10)
except Exception:
    pass
layout["Music_Hall"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/11_icon_Just_Announced_by_My_Performers.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 1688), _c11)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/13_icon_Dallas_Mavericks.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 975), _c13)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/14_icon_Music_Hall.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 807), _c14)
except Exception:
    pass
layout["Music_Hall"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 68)
    canvas.paste(_c15, (1319, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 59, 64)
    canvas.paste(_c17, (313, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [313, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/18_icon_8.35_my.png
try:
    _c18 = get_crop(18, 47, 64)
    canvas.paste(_c18, (186, 1), _c18)
except Exception:
    pass
layout["8.35_my"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/19_icon_Events_by_My_Performers.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 1520), _c19)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/20_icon_Search.png
try:
    _c20 = get_crop(20, 288, 162)
    canvas.paste(_c20, (288, 2792), _c20)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/21_icon_Radio.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 807), _c21)
except Exception:
    pass
layout["Radio"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/22_icon_Dallas_Mavericks.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/24_icon_Performer_event_or_venue.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/25_icon_8.35_my.png
try:
    _c25 = get_crop(25, 57, 64)
    canvas.paste(_c25, (114, 1), _c25)
except Exception:
    pass
layout["8.35_my"] = [114, 1, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 168, 144)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/27_text_Suggestions.png
try:
    _c27 = get_crop(27, 331, 74)
    canvas.paste(_c27, (40, 1423), _c27)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_09_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-12/28_text_Just_Announced_by_My_Performers.png
try:
    _c28 = get_crop(28, 1440, 168)
    canvas.paste(_c28, (0, 1856), _c28)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]
