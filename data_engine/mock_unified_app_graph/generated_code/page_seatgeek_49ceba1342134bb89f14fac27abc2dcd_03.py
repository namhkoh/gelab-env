# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_03
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6.png
# step_index: 3/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# top status bar
draw.rectangle([(0, 0), (1440, 84)], fill="#f3f3f3")
draw.line([(0, 84), (1440, 84)], fill="#e1e1e1", width=1)

# search bar background (rounded)
search_left, search_top = 40, 48
search_right, search_bottom = 1400, 192
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=28,
    fill="#fafafa",
    outline="#e9e9e9",
    width=1
)
# subtle shadow under search bar
draw.line([(search_left+4, search_bottom+6), (search_right-4, search_bottom+6)], fill="#f0f0f0", width=6)

# divider below header/search
draw.line([(24, 204), (1440-24, 204)], fill="#ececec", width=1)

# subtle section separators
draw.line([(24, 1360), (1440-24, 1360)], fill="#ececec", width=1)
draw.line([(24, 1408), (1440-24, 1408)], fill="#f5f5f5", width=1)

# suggestions section background band (very subtle)
draw.rectangle([(0, 1408), (1440, 2048)], fill="#ffffff")

# faint divider between recent searches and suggestions area
draw.line([(24, 1288), (1440-24, 1288)], fill="#f2f2f2", width=1)

# bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
draw.line([(0, nav_top), (1440, nav_top)], fill="#e9e9e9", width=1)
# small soft shadow above nav bar
draw.rectangle([(0, nav_top-6), (1440, nav_top)], fill="#fbfbfb")

# large page background (ensure consistent white)
draw.rectangle([(0, 0), (1440, 2960)], outline=None, fill="#ffffff")

# subtle vertical guide lines (very light) to match UI spacing (non-intrusive)
for x in (24, 40, 1400, 1416):
    draw.line([(x, 204), (x, 2600)], fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/00_icon_Shin_Lim.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 807), _c0)
except Exception:
    pass
layout["Shin_Lim"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/01_icon_8.34_my.png
try:
    _c1 = get_crop(1, 168, 144)
    canvas.paste(_c1, (48, 120), _c1)
except Exception:
    pass
layout["8.34_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 47, 70)
    canvas.paste(_c2, (1153, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/03_icon_Tracking.png
try:
    _c3 = get_crop(3, 288, 168)
    canvas.paste(_c3, (864, 2792), _c3)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/04_icon_Browse.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (0, 2792), _c4)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/05_icon_WWE.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 975), _c5)
except Exception:
    pass
layout["WWE"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/06_icon_Music_Hall.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 471), _c6)
except Exception:
    pass
layout["Music_Hall"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 62, 64)
    canvas.paste(_c7, (243, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [243, 2, 305, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/08_icon_Radio.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 639), _c8)
except Exception:
    pass
layout["Radio"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/10_icon_Just_Announced_by_My_Performers.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1688), _c10)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 96, 68)
    canvas.paste(_c11, (1216, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1216, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/13_icon_WWE.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 1143), _c13)
except Exception:
    pass
layout["WWE"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/14_icon_8.34_my.png
try:
    _c14 = get_crop(14, 53, 64)
    canvas.paste(_c14, (116, 1), _c14)
except Exception:
    pass
layout["8.34_my"] = [116, 1, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/15_icon_Dallas_Mavericks.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 975), _c15)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/16_icon_8.34_my.png
try:
    _c16 = get_crop(16, 47, 63)
    canvas.paste(_c16, (186, 1), _c16)
except Exception:
    pass
layout["8.34_my"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/17_icon_8.34_my.png
try:
    _c17 = get_crop(17, 91, 63)
    canvas.paste(_c17, (16, 3), _c17)
except Exception:
    pass
layout["8.34_my"] = [16, 3, 107, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/19_icon_Dallas_Mavericks.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 807), _c19)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 59, 64)
    canvas.paste(_c20, (313, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [313, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 53, 68)
    canvas.paste(_c21, (1319, 0), _c21)
except Exception:
    pass
layout["icon_21"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/22_icon_Events_by_My_Performers.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1520), _c22)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/23_icon_Music_Hall.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 639), _c23)
except Exception:
    pass
layout["Music_Hall"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/25_icon_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/26_icon_Search.png
try:
    _c26 = get_crop(26, 288, 162)
    canvas.paste(_c26, (288, 2792), _c26)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/28_text_WWE.png
try:
    _c28 = get_crop(28, 127, 45)
    canvas.paste(_c28, (239, 1204), _c28)
except Exception:
    pass
layout["WWE"] = [239, 1204, 366, 1249]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/29_text_Suggestions.png
try:
    _c29 = get_crop(29, 331, 74)
    canvas.paste(_c29, (40, 1423), _c29)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_03_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-6/30_text_Just_Announced_by_My_Performers.png
try:
    _c30 = get_crop(30, 1440, 168)
    canvas.paste(_c30, (0, 1856), _c30)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]
