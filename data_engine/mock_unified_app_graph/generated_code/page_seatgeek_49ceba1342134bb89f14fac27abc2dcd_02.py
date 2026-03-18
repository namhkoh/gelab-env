# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_02
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5.png
# step_index: 2/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a very light off-white
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")

# Status bar (top area)
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill="#EEEEEE")
# subtle bottom hairline under status bar
draw.line([(0, status_h-1), (1440, status_h-1)], fill="#E0E0E0", width=1)

# Search bar background (rounded)
search_left = 48
search_top = 104
search_right = 1392
search_bottom = 256
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=28,
    fill="#F6F6F6",
    outline="#E6E6E6",
    width=1
)

# Thin divider under search area
divider_y = search_bottom + 12
draw.line([(40, divider_y), (1400, divider_y)], fill="#E9E9E9", width=1)

# Large subtle white card grouping recent searches and list area
card_left = 24
card_top = divider_y + 18
card_right = 1416
card_bottom = 1360
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=18,
    fill="#FFFFFF",
    outline="#EFEFEF",
    width=1
)
# slight inner top hairline to separate from search divider
draw.line([(card_left+8, card_top+2), (card_right-8, card_top+2)], fill="#F5F5F5", width=1)

# Section separator between the list block and Suggestions (faint)
suggestion_sep_y = 1320
draw.line([(40, suggestion_sep_y), (1400, suggestion_sep_y)], fill="#F0F0F0", width=1)

# Suggestions block background (subtle, but keep very similar to page to avoid duplicating icons/text)
suggestions_top = suggestion_sep_y + 24
suggestions_bottom = suggestions_top + 420
draw.rectangle([(24, suggestions_top), (1416, suggestions_bottom)], fill="#FFFFFF")
# faint left margin guide line (decorative structure)
draw.line([(24, suggestions_top+8), (24, suggestions_bottom-8)], fill="#FFFFFF", width=6)

# Footer / bottom navigation background
nav_top = 2760
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
# top border/shadow for the nav area
draw.line([(0, nav_top), (1440, nav_top)], fill="#E9E9E9", width=1)
# subtle elevated center area behind nav icons (rounded pill)
nav_center_left = 240
nav_center_right = 1200
nav_center_top = nav_top + 22
nav_center_bottom = nav_top + 88
draw.rounded_rectangle(
    [(nav_center_left, nav_center_top), (nav_center_right, nav_center_bottom)],
    radius=34,
    fill="#FFFFFF",
    outline="#FFFFFF"
)

# Additional subtle global separators to match structured layout
# underline for the search card top edge to emphasize separation
draw.line([(48, card_top - 6), (1392, card_top - 6)], fill="#F7F7F7", width=1)
# faint vertical guide on the left to align content groups (purely structural)
draw.line([(40, card_top), (40, card_bottom)], fill="#FFFFFF", width=6)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/00_icon_Shin_Lim.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 807), _c0)
except Exception:
    pass
layout["Shin_Lim"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/01_icon_Radio.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["Radio"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 50, 69)
    canvas.paste(_c2, (1152, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1152, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/03_icon_Tracking.png
try:
    _c3 = get_crop(3, 288, 168)
    canvas.paste(_c3, (864, 2792), _c3)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 64, 65)
    canvas.paste(_c4, (242, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [242, 2, 306, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/05_icon_Browse.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (0, 2792), _c5)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 98, 69)
    canvas.paste(_c6, (1215, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/07_icon_8.34_Wy.png
try:
    _c7 = get_crop(7, 54, 65)
    canvas.paste(_c7, (115, 0), _c7)
except Exception:
    pass
layout["8.34_Wy"] = [115, 0, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/08_icon_Tickets.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (576, 2792), _c8)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/09_icon_WWE.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 975), _c9)
except Exception:
    pass
layout["WWE"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/10_icon_Just_Announced_by_My_Performers.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1688), _c10)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/11_icon_Music_Hall.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 471), _c11)
except Exception:
    pass
layout["Music_Hall"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/12_icon_8.34_Wy.png
try:
    _c12 = get_crop(12, 94, 65)
    canvas.paste(_c12, (13, 0), _c12)
except Exception:
    pass
layout["8.34_Wy"] = [13, 0, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/13_icon_8.34_Wy.png
try:
    _c13 = get_crop(13, 48, 64)
    canvas.paste(_c13, (185, 1), _c13)
except Exception:
    pass
layout["8.34_Wy"] = [185, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/15_icon_8.34_Wy.png
try:
    _c15 = get_crop(15, 168, 144)
    canvas.paste(_c15, (48, 120), _c15)
except Exception:
    pass
layout["8.34_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 52, 68)
    canvas.paste(_c16, (1319, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/17_icon_Dallas_Mavericks.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 975), _c17)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 62, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 375, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/20_icon_Dallas_Mavericks.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 807), _c20)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/21_icon_Events_by_My_Performers.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 1520), _c21)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/22_icon_WWE.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["WWE"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/23_icon_Music_Hall.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 639), _c23)
except Exception:
    pass
layout["Music_Hall"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/25_icon_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/26_icon_Just_Announced_by_My_Performers.png
try:
    _c26 = get_crop(26, 1440, 168)
    canvas.paste(_c26, (0, 1856), _c26)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/27_icon_Dallas_Mavericks.png
try:
    _c27 = get_crop(27, 1440, 168)
    canvas.paste(_c27, (0, 1143), _c27)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/28_text_Recent_searches.png
try:
    _c28 = get_crop(28, 168, 144)
    canvas.paste(_c28, (48, 120), _c28)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_02_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-5/29_text_Suggestions.png
try:
    _c29 = get_crop(29, 331, 74)
    canvas.paste(_c29, (40, 1423), _c29)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
