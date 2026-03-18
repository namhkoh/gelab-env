# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_08
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11.png
# step_index: 8/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# background fill
bg_color = (255, 255, 255)
status_bar_color = (242, 242, 242)
search_bg = (246, 246, 247)
divider_color = (224, 224, 224)
card_bg = (250, 250, 250)
nav_shadow = (230, 230, 230)

# fill whole canvas (ensure consistent base)
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# status bar area (approx 56px high)
draw.rectangle([(0, 0), (1440, 56)], fill=status_bar_color)

# large search bar rounded rectangle (matches detected search crop area)
search_left, search_top = 48, 48
search_right, search_bottom = 1392, 192  # height 144 as in detection
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=24,
    fill=search_bg,
    outline=None,
)

# subtle inner highlight on top edge of search bar
draw.line([(search_left + 2, search_top + 2), (search_right - 2, search_top + 2)], fill=(255,255,255), width=1)

# thin divider below search area
divider_y = search_bottom + 8
draw.line([(48, divider_y), (1392, divider_y)], fill=divider_color, width=2)

# horizontal separator between Recent searches and Suggestions
sep_y = 1120
draw.line([(48, sep_y), (1392, sep_y)], fill=divider_color, width=2)

# subtle background card for Suggestions area (rounded)
suggestions_top = 1320
suggestions_bottom = 1700
draw.rounded_rectangle(
    [(48, suggestions_top), (1392, suggestions_bottom)],
    radius=12,
    fill=card_bg,
    outline=None,
)

# internal subtle separators for suggestion rows (approx positions)
draw.line([(84, suggestions_top + 120), (1360, suggestions_top + 120)], fill=(242,242,242), width=1)
draw.line([(84, suggestions_top + 240), (1360, suggestions_top + 240)], fill=(242,242,242), width=1)

# faint large card/background behind Recent searches list (very subtle)
recent_top = 220
recent_bottom = 1240
draw.rounded_rectangle(
    [(36, recent_top), (1404, recent_bottom)],
    radius=6,
    fill=(255,255,255),
    outline=None,
)

# subtle top/bottom shadows to separate content sections
draw.line([(36, recent_top), (1404, recent_top)], fill=(245,245,245), width=1)
draw.line([(36, recent_bottom), (1404, recent_bottom)], fill=(245,245,245), width=1)

# bottom navigation bar separator/shadow (nav area starts at 2792 height)
nav_top = 2792
draw.line([(0, nav_top), (1440, nav_top)], fill=nav_shadow, width=2)
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255,255,255))

# small subtle left and right padding guides (visual only, very faint)
draw.line([(48, 0), (48, 2960)], fill=(255,255,255), width=1)
draw.line([(1392, 0), (1392, 2960)], fill=(255,255,255), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 49, 70)
    canvas.paste(_c0, (1152, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1152, 0, 1201, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/01_icon_Suggestions.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 1143), _c1)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/02_icon_Tracking.png
try:
    _c2 = get_crop(2, 288, 168)
    canvas.paste(_c2, (864, 2792), _c2)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 65)
    canvas.paste(_c3, (242, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [242, 2, 306, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 98, 69)
    canvas.paste(_c4, (1215, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/05_icon_Browse.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (0, 2792), _c5)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/06_icon_Shin_Lim.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 975), _c6)
except Exception:
    pass
layout["Shin_Lim"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/07_icon_Music_Hall.png
try:
    _c7 = get_crop(7, 1440, 168)
    canvas.paste(_c7, (0, 471), _c7)
except Exception:
    pass
layout["Music_Hall"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/08_icon_Tickets.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (576, 2792), _c8)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/09_icon_Music_Hall.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 639), _c9)
except Exception:
    pass
layout["Music_Hall"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/10_icon_Just_Announced_by_My_Performers.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1688), _c10)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/11_icon_Clear.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 120), _c11)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 69)
    canvas.paste(_c12, (1319, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1319, 0, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/13_icon_8.35_my.png
try:
    _c13 = get_crop(13, 48, 64)
    canvas.paste(_c13, (185, 1), _c13)
except Exception:
    pass
layout["8.35_my"] = [185, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/14_icon_Account.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (1152, 2792), _c14)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 61, 63)
    canvas.paste(_c15, (313, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [313, 2, 374, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/16_icon_Music_Hall.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 807), _c16)
except Exception:
    pass
layout["Music_Hall"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/17_icon_Dallas_Mavericks.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 975), _c17)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/18_icon_8.35_my.png
try:
    _c18 = get_crop(18, 168, 144)
    canvas.paste(_c18, (48, 120), _c18)
except Exception:
    pass
layout["8.35_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/19_icon_8.35_my.png
try:
    _c19 = get_crop(19, 58, 65)
    canvas.paste(_c19, (113, 0), _c19)
except Exception:
    pass
layout["8.35_my"] = [113, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/20_icon_Events_by_My_Performers.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 1520), _c20)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/21_icon_Radio.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 807), _c21)
except Exception:
    pass
layout["Radio"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/22_icon_Dallas_Mavericks.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/24_icon_Performer_event_or_venue.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/25_icon_Recent_searches.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 471), _c25)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/26_icon_Just_Announced_by_My_Performers.png
try:
    _c26 = get_crop(26, 1440, 168)
    canvas.paste(_c26, (0, 1856), _c26)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_08_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-11/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
