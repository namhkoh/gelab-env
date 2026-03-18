# page_id: page_eventbrite_66847fb559f849b19cea93b83307fae7_01
# screenshot: 2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3.png
# step_index: 1/4
# task: Open Eventbrite. Open favorites and select the second event. Process to checkout and see what payment options it offers.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (dominant color is white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area at top (~56px tall) - light grey/neutral to match screenshot
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#BDBDBD")

# Top toolbar area (under status bar) - keep white but add subtle bottom divider
toolbar_top = status_h
toolbar_bottom = 140
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill="#FFFFFF")
draw.line([(0, toolbar_bottom), (1440, toolbar_bottom)], fill="#EDEBF0", width=1)

# Search field background (rounded) — use detected position (do not draw icons/text inside)
search_x, search_y = 195, 93
search_w, search_h = 1179, 144
search_rect = [search_x, search_y, search_x + search_w, search_y + search_h]
# subtle drop shadow for the search bar
shadow_offset = 6
draw.rounded_rectangle(
    [search_rect[0], search_rect[1] + shadow_offset, search_rect[2], search_rect[3] + shadow_offset],
    radius=72, fill="#F3F3F5"
)
# search field
draw.rounded_rectangle(search_rect, radius=72, fill="#FFFFFF", outline="#E7E1F2", width=4)

# Section cards (rounded rectangles behind each event group)
card_x = 48
card_w = 1344
# y positions and heights taken from detected elements (do not draw text/images/icons)
rows = [
    (490, 396),   # first row
    (886, 396),   # second
    (1282, 396),  # third
    (1678, 396),  # fourth
    (2074, 396),  # fifth
    (2470, 346),  # sixth (slightly shorter)
]
for (y, h) in rows:
    x1, y1 = card_x, y
    x2, y2 = card_x + card_w, y + h
    # light shadow under the card
    draw.rounded_rectangle([x1 + 0, y1 + 8, x2 + 0, y2 + 8], radius=16, fill="#F2F2F4")
    # actual card background (very subtle off-white)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=16, fill="#FFFFFF", outline="#EFEFF1", width=1)
    # thin separator line below each card
    draw.line([(x1 + 12, y2 + 12), (x2 - 12, y2 + 12)], fill="#F0EFF1", width=1)

# Additional subtle separators between major sections (to match screenshot rhythm)
# e.g., separator below the "More events you'll love" title area (around y ~440)
draw.line([(48, 440), (1392, 440)], fill="#F1F0F3", width=1)

# Bottom navigation bar background and top divider (detected nav area height ~156 at y=2804)
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#FFFFFF")
draw.line([(0, nav_top), (1440, nav_top)], fill="#EDEBF0", width=1)

# Subtle elevated card for the floating location pill area (leave actual pill untouched;
# only draw a soft background layer behind it to match screenshot layering)
# Detected floating pill is at (492,2651) size (456x117) - draw a soft shadow behind but not the pill itself
pill_x, pill_y, pill_w, pill_h = 492, 2651, 456, 117
draw.rounded_rectangle(
    [pill_x - 8, pill_y + 8, pill_x + pill_w + 8, pill_y + pill_h + 8],
    radius=40, fill="#F5F5F7"
)

# Final very light overall vertical guideline to visually group content (subtle)
draw.line([(48, 460), (48, nav_top - 180)], fill="#FAFAFB", width=2)
draw.line([(1392, 460), (1392, nav_top - 180)], fill="#FAFAFB", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/00_icon_Free.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["Free"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/01_icon_NDIE_DANCEPA.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/02_icon_Ibaigktsinel.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1282), _c2)
except Exception:
    pass
layout["Ibaigktsinel"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/04_icon_Los_Angeles.png
try:
    _c4 = get_crop(4, 456, 117)
    canvas.paste(_c4, (492, 2651), _c4)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/05_icon_NDIE.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 886), _c5)
except Exception:
    pass
layout["NDIE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/06_icon_Afliccion_Perdida_y.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/07_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 490), _c7)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1140, 1555), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 1935), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1284, 1555), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/11_icon_Afliccion_Perdida_y.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1140, 2347), _c11)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 2347), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1143), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/14_icon_Club_Decades.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1140, 1143), _c14)
except Exception:
    pass
layout["Club_Decades"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/15_icon_The_Gr.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["The_Gr"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/16_icon_Favorite_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1140, 763), _c16)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1284, 763), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 59)
    canvas.paste(_c18, (311, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [311, 3, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/19_icon_7.38.png
try:
    _c19 = get_crop(19, 58, 61)
    canvas.paste(_c19, (181, 2), _c19)
except Exception:
    pass
layout["7.38"] = [181, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/20_icon_Home.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 52, 60)
    canvas.paste(_c21, (247, 2), _c21)
except Exception:
    pass
layout["icon_21"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/22_icon_8_4717_creator_followers.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 886), _c22)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/23_icon_7.38.png
try:
    _c23 = get_crop(23, 103, 99)
    canvas.paste(_c23, (41, 122), _c23)
except Exception:
    pass
layout["7.38"] = [41, 122, 144, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/24_icon_59_creator_followers.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1678), _c24)
except Exception:
    pass
layout["59_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 48, 53)
    canvas.paste(_c25, (1320, 7), _c25)
except Exception:
    pass
layout["icon_25"] = [1320, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 92, 58)
    canvas.paste(_c26, (1212, 4), _c26)
except Exception:
    pass
layout["icon_26"] = [1212, 4, 1304, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/27_icon_7.38.png
try:
    _c27 = get_crop(27, 60, 62)
    canvas.paste(_c27, (114, 1), _c27)
except Exception:
    pass
layout["7.38"] = [114, 1, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/28_icon_Public_House_Los_Angeles_CA.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 490), _c28)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/29_icon_8_21119_creator_followers.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 1282), _c29)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 44, 57)
    canvas.paste(_c30, (385, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/31_icon_Grief_Loss_Resiliency.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2074), _c31)
except Exception:
    pass
layout["Grief;_Loss,_Resiliency"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/32_icon_9.30_PM_PDT.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["9.30_PM_PDT"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/33_icon_Free.png
try:
    _c33 = get_crop(33, 127, 74)
    canvas.paste(_c33, (246, 1748), _c33)
except Exception:
    pass
layout["Free"] = [246, 1748, 373, 1822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/34_icon_YEAH_YEAH_YAS_Queer_Indie_Dance_Party_LA.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["YEAH_YEAH_YAS:_Queer_Indi"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/35_icon_8_21119_creator_followers.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1282), _c35)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/36_icon_8_4717_creator_followers.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 886), _c36)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/37_icon_icon_37.png
try:
    _c37 = get_crop(37, 41, 53)
    canvas.paste(_c37, (1272, 7), _c37)
except Exception:
    pass
layout["icon_37"] = [1272, 7, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/38_text_7.38.png
try:
    _c38 = get_crop(38, 92, 43)
    canvas.paste(_c38, (22, 17), _c38)
except Exception:
    pass
layout["7.38"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/40_text_2000s_NITE.png
try:
    _c40 = get_crop(40, 202, 49)
    canvas.paste(_c40, (81, 2528), _c40)
except Exception:
    pass
layout["2000s_NITE"] = [81, 2528, 283, 2577]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/41_text_Fri_May_31.png
try:
    _c41 = get_crop(41, 184, 43)
    canvas.paste(_c41, (392, 2525), _c41)
except Exception:
    pass
layout["Fri,_May_31"] = [392, 2525, 576, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/42_text_9_30_PM_PDT.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["9:30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/43_text_UNDERGROUND_X_200Os_NITE_Dance_Partyl.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["UNDERGROUND_X_200Os_NITE_"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/44_text_MEt.png
try:
    _c44 = get_crop(44, 79, 67)
    canvas.paste(_c44, (205, 2598), _c44)
except Exception:
    pass
layout["MEt"] = [205, 2598, 284, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/45_clickable_Favorites.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (576, 2804), _c45)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/46_clickable_Tickets.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (864, 2804), _c46)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_01_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-3/47_clickable_More.png
try:
    _c47 = get_crop(47, 288, 156)
    canvas.paste(_c47, (1152, 2804), _c47)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
