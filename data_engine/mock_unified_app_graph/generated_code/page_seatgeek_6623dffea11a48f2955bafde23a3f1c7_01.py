# page_id: page_seatgeek_6623dffea11a48f2955bafde23a3f1c7_01
# screenshot: 2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4.png
# step_index: 1/9
# task: Open SeatGeek. Search "New York Knicks" and select the second upcoming event, show the location of the event and track the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already white) - add a very subtle off-white tint for the main background
draw.rectangle([(0, 0), canvas.size], fill="#fcfcfd")

# Status bar area (top ~80px) - light gray bar to match screenshot
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#efefef")

# Header area under status bar
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# thin divider under header
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill="#e6e6e6", width=1)

# Big featured card background (rounded rectangle) behind the "New York Knicks" card
card_left, card_top = 48, 360
card_right, card_bottom = 48 + 1344, 360 + 840
card_radius = 28

# Create a vertical gradient for the big blue card (top -> bottom)
top_color = (41, 132, 196)   # #2984c4
bottom_color = (5, 85, 140)  # #05558c
card_height = card_bottom - card_top
for i in range(card_height):
    t = i / max(1, card_height - 1)
    r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
    g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
    b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
    draw.line([(card_left, card_top + i), (card_right, card_top + i)], fill=(r, g, b))

# rounded clip-like rectangle outline (subtle darker border)
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                       radius=card_radius, outline="#0e4f80", width=2)

# "Just for you" section - background is same as canvas; draw a subtle container for the row of cards
just_row_top = 1431 - 16
just_row_bottom = 1950 + 16
just_row_left = 24
just_row_right = 1440 - 24
draw.rectangle([(just_row_left, just_row_top), (just_row_right, just_row_bottom)], fill="#ffffff")
# subtle shadow line above the row
draw.line([(just_row_left, just_row_top), (just_row_right, just_row_top)], fill="#fbfbfb", width=2)

# Individual card backgrounds (rounded) for the three thumbnails (these are backgrounds only)
cards = [
    (48, 1431, 48 + 462, 1431 + 519),
    (546, 1431, 546 + 462, 1431 + 519),
    (1044, 1431, 1044 + 396, 1431 + 519)
]
for (l, t, r, b) in cards:
    # Draw a subtle rounded card container background (very light gray)
    draw.rounded_rectangle([(l, t), (r, b)], radius=18, fill="#fbfbfb", outline="#ececec", width=1)

# Divider below the "Just for you" row
divider_y = just_row_bottom + 28
draw.line([(24, divider_y), (1440 - 24, divider_y)], fill="#e6e6e6", width=1)

# Trending events container background (slightly warm white)
trending_top = divider_y + 28
trending_left = 24
trending_right = 1440 - 24
trending_bottom = 2720  # leave space above bottom nav
draw.rectangle([(trending_left, trending_top), (trending_right, trending_bottom)], fill="#ffffff")

# Draw separators for trending list items (approximate positions taken from detected items)
# First trending item spans roughly y=2183..2419 (detected); draw separators at bottoms of list rows
sep_positions = [2419, 2655]
for y in sep_positions:
    draw.line([(trending_left + 20, y), (trending_right - 20, y)], fill="#efefef", width=1)

# Add faint horizontal separators for visual grouping (between header and list)
draw.line([(trending_left, trending_top), (trending_right, trending_top)], fill="#fbfbfb", width=2)

# Bottom navigation bar background and top divider/shadow
nav_top = 2792
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#ffffff")
# subtle top shadow line
draw.line([(0, nav_top), (1440, nav_top)], fill="#e8e8e8", width=2)

# Small rounded pill behind the active tab indicator area (leftmost) to mimic subtle background
pill_margin = 24
draw.rounded_rectangle([(pill_margin, nav_top + 12), (pill_margin + 156, nav_bottom - 12)],
                       radius=40, fill="#ffffff", outline="#fafafa", width=1)

# Final subtle overall vignette/edge shading to match screenshot depth (very light)
edge_shade = (0, 0, 0, 12)
# top shadow under status/header (soft)
draw.line([(0, header_bottom), (1440, header_bottom)], fill="#f1f1f1", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/00_icon_Knicks.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Knicks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/01_icon_BOOK_OF.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (48, 1431), _c1)
except Exception:
    pass
layout["BOOK_OF"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/02_icon_August_Wilson_Theatre.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 2183), _c2)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/03_icon_Yankee_Stadium.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 2419), _c3)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/04_icon_S116.png
try:
    _c4 = get_crop(4, 396, 519)
    canvas.paste(_c4, (1044, 1431), _c4)
except Exception:
    pass
layout["S116+"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/05_icon_S94.png
try:
    _c5 = get_crop(5, 462, 519)
    canvas.paste(_c5, (546, 1431), _c5)
except Exception:
    pass
layout["S94+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 99, 152)
    canvas.paste(_c6, (1341, 2464), _c6)
except Exception:
    pass
layout["icon_6"] = [1341, 2464, 1440, 2616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/07_icon_View_all.png
try:
    _c7 = get_crop(7, 98, 149)
    canvas.paste(_c7, (1342, 2228), _c7)
except Exception:
    pass
layout["View_all"] = [1342, 2228, 1440, 2377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/08_icon_May.png
try:
    _c8 = get_crop(8, 1309, 236)
    canvas.paste(_c8, (0, 2183), _c8)
except Exception:
    pass
layout["May"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/09_icon_New_York_NY.png
try:
    _c9 = get_crop(9, 62, 58)
    canvas.paste(_c9, (243, 5), _c9)
except Exception:
    pass
layout["New_York,_NY"] = [243, 5, 305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/10_icon_6.57_Wy.png
try:
    _c10 = get_crop(10, 56, 56)
    canvas.paste(_c10, (114, 6), _c10)
except Exception:
    pass
layout["6.57_Wy"] = [114, 6, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/11_icon_888.png
try:
    _c11 = get_crop(11, 99, 65)
    canvas.paste(_c11, (1214, 0), _c11)
except Exception:
    pass
layout["888"] = [1214, 0, 1313, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/12_icon_6.57_Wy.png
try:
    _c12 = get_crop(12, 50, 56)
    canvas.paste(_c12, (184, 6), _c12)
except Exception:
    pass
layout["6.57_Wy"] = [184, 6, 234, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/13_icon_888.png
try:
    _c13 = get_crop(13, 144, 240)
    canvas.paste(_c13, (1260, 72), _c13)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/14_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (864, 2792), _c14)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/15_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (288, 2792), _c15)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/16_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (576, 2792), _c16)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 64)
    canvas.paste(_c17, (1319, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 1, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 47, 66)
    canvas.paste(_c18, (1154, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1154, 0, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 99, 119)
    canvas.paste(_c19, (1341, 2698), _c19)
except Exception:
    pass
layout["icon_19"] = [1341, 2698, 1440, 2817]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/20_icon_Browse.png
try:
    _c20 = get_crop(20, 288, 162)
    canvas.paste(_c20, (0, 2792), _c20)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/21_icon_Account.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (1152, 2792), _c21)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 53, 59)
    canvas.paste(_c22, (316, 5), _c22)
except Exception:
    pass
layout["icon_22"] = [316, 5, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/23_icon_Andrew_Schulz.png
try:
    _c23 = get_crop(23, 462, 519)
    canvas.paste(_c23, (546, 1431), _c23)
except Exception:
    pass
layout["Andrew_Schulz"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 116, 127)
    canvas.paste(_c24, (1138, 2484), _c24)
except Exception:
    pass
layout["icon_24"] = [1138, 2484, 1254, 2611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/25_icon_New_York_NY.png
try:
    _c25 = get_crop(25, 390, 87)
    canvas.paste(_c25, (40, 119), _c25)
except Exception:
    pass
layout["New_York,_NY"] = [40, 119, 430, 206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/26_icon_The.png
try:
    _c26 = get_crop(26, 91, 102)
    canvas.paste(_c26, (36, 1427), _c26)
except Exception:
    pass
layout["The"] = [36, 1427, 127, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/27_text_date.png
try:
    _c27 = get_crop(27, 114, 52)
    canvas.paste(_c27, (137, 208), _c27)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/28_text_Just_for_you.png
try:
    _c28 = get_crop(28, 306, 66)
    canvas.paste(_c28, (38, 1310), _c28)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/29_text_View_all.png
try:
    _c29 = get_crop(29, 264, 183)
    canvas.paste(_c29, (1176, 1248), _c29)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/30_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c30 = get_crop(30, 288, 168)
    canvas.paste(_c30, (576, 2792), _c30)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (408, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_01_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-4/32_clickable_Tracking.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (906, 1455), _c32)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
