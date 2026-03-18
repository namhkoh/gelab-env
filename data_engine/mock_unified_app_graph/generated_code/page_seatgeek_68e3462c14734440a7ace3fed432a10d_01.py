# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_01
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4.png
# step_index: 1/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")

# Status bar (top area)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#F3F3F3")
# subtle bottom hairline for status bar
draw.line([(0, status_h-1), (1440, status_h-1)], fill="#E2E2E2", width=1)

# Header / toolbar area (location + date area)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# subtle divider under header
draw.line([(32, header_bottom-1), (1408, header_bottom-1)], fill="#E8E8E8", width=1)

# Main content background card (large white sheet to hold sections)
content_top = header_bottom
content_bottom = 2620
content_left = 24
content_right = 1416
draw.rounded_rectangle(
    [(content_left, content_top), (content_right, content_bottom)],
    radius=24,
    fill="#FFFFFF",
    outline=None
)

# Gentle drop shadow under the main content card (thin)
shadow_y = content_bottom
for i, a in enumerate([180, 140, 100, 60], start=1):
    alpha = int(a * 0.01 * 255 / 2)
    # We can't add semi-transparent easily without imports; emulate with faint lines
    draw.line([(content_left + 6, shadow_y + i), (content_right - 6, shadow_y + i)], fill="#F0F0F0", width=1)

# Divider lines separating major sections (do not draw text/icons)
# Under large banner area (banner will be pasted on top; we provide a divider beneath)
banner_bottom = 1208  # approx bottom of big banner card
draw.line([(48, banner_bottom), (1392, banner_bottom)], fill="#EFEFEF", width=1)

# "Just for you" section divider (top and bottom hairlines for the row)
just_top = 1310
just_bottom = 1430
draw.line([(32, just_top-20), (1408, just_top-20)], fill="#FFFFFF", width=1)  # subtle spacing
draw.line([(32, just_bottom+520), (1408, just_bottom+520)], fill="#F1F1F1", width=1)  # below cards area

# Trending events header area divider
trending_top = 1630
draw.line([(32, trending_top), (1408, trending_top)], fill="#F5F5F5", width=1)
# Multiple thin separators for trending list items
sep_positions = [2000, 2280, 2560]  # approximate y positions for separators
for y in sep_positions:
    draw.line([(32, y), (1408, y)], fill="#EFEFEF", width=1)

# Light left/right gutters to mirror app layout (subtle)
gutter_color = "#FBFBFB"
draw.rectangle([(0, 0), (24, 2960)], fill=gutter_color)
draw.rectangle([(1416, 0), (1440, 2960)], fill=gutter_color)

# Bottom navigation bar background and top divider/shadow
nav_top = 2680
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#FFFFFF")
# thin top border shadow above nav
draw.line([(24, nav_top), (1416, nav_top)], fill="#E6E6E6", width=1)
# slight inner shadow band
draw.line([(24, nav_top+2), (1416, nav_top+2)], fill="#F8F8F8", width=1)

# Side peek cards / carousel background (right edge cards area)
# Provide a subtle rounded mask on the right to indicate cropping of carousel
carousel_right_edge = 1408
draw.rectangle([(1376, 360), (1440, 1200)], fill="#FFFFFF")
draw.line([(1398, 360), (1398, 1200)], fill="#F0F0F0", width=1)

# Accent rounded pill behind filter icon area (top-right) - only background shape
filter_box = (1320, 96, 1410, 176)
draw.rounded_rectangle(filter_box, radius=12, fill="#FFFFFF", outline="#EDEDED", width=1)

# Final subtle overall vignette lines (very faint to match screenshot's polished look)
draw.line([(32, 300), (1408, 300)], fill="#FAFAFA", width=1)
draw.line([(32, 880), (1408, 880)], fill="#FAFAFA", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/00_icon_St._James_Theatre.png
try:
    _c0 = get_crop(0, 1309, 236)
    canvas.paste(_c0, (0, 2183), _c0)
except Exception:
    pass
layout["St._James_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 100, 151)
    canvas.paste(_c1, (1340, 2464), _c1)
except Exception:
    pass
layout["icon_1"] = [1340, 2464, 1440, 2615]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 100, 147)
    canvas.paste(_c2, (1340, 2228), _c2)
except Exception:
    pass
layout["View_all"] = [1340, 2228, 1440, 2375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/03_icon_840.png
try:
    _c3 = get_crop(3, 96, 63)
    canvas.paste(_c3, (1217, 1), _c3)
except Exception:
    pass
layout["840"] = [1217, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/04_icon_BARCLAYS_CEME.png
try:
    _c4 = get_crop(4, 462, 519)
    canvas.paste(_c4, (48, 1431), _c4)
except Exception:
    pass
layout["BARCLAYS_CEME"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/05_icon_S87.png
try:
    _c5 = get_crop(5, 396, 519)
    canvas.paste(_c5, (1044, 1431), _c5)
except Exception:
    pass
layout["S87+"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/06_icon_NCAA_M_Basketball_Brooklyn.png
try:
    _c6 = get_crop(6, 1309, 236)
    canvas.paste(_c6, (0, 2419), _c6)
except Exception:
    pass
layout["NCAA_M_Basketball_Brookly"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/07_icon_TIcKETS.png
try:
    _c7 = get_crop(7, 1344, 840)
    canvas.paste(_c7, (48, 360), _c7)
except Exception:
    pass
layout["TIcKETS"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/08_icon_8.30_my.png
try:
    _c8 = get_crop(8, 56, 59)
    canvas.paste(_c8, (182, 3), _c8)
except Exception:
    pass
layout["8.30_my"] = [182, 3, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 61)
    canvas.paste(_c9, (1320, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1320, 3, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/10_icon_840.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (1260, 72), _c10)
except Exception:
    pass
layout["840"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/11_icon_8.30_my.png
try:
    _c11 = get_crop(11, 51, 58)
    canvas.paste(_c11, (116, 3), _c11)
except Exception:
    pass
layout["8.30_my"] = [116, 3, 167, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 51, 55)
    canvas.paste(_c12, (316, 6), _c12)
except Exception:
    pass
layout["icon_12"] = [316, 6, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 45, 64)
    canvas.paste(_c13, (1155, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1155, 1, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/14_icon_New_York_NY.png
try:
    _c14 = get_crop(14, 52, 57)
    canvas.paste(_c14, (247, 5), _c14)
except Exception:
    pass
layout["New_York,_NY"] = [247, 5, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/15_icon_S2_4_D.png
try:
    _c15 = get_crop(15, 127, 154)
    canvas.paste(_c15, (1134, 2467), _c15)
except Exception:
    pass
layout["S2_(#4_D="] = [1134, 2467, 1261, 2621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/16_icon_Hadestown.png
try:
    _c16 = get_crop(16, 288, 162)
    canvas.paste(_c16, (0, 2792), _c16)
except Exception:
    pass
layout["Hadestown"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/17_icon_Tracking.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (864, 2792), _c17)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 99, 107)
    canvas.paste(_c18, (1341, 2699), _c18)
except Exception:
    pass
layout["icon_18"] = [1341, 2699, 1440, 2806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/19_icon_S86.png
try:
    _c19 = get_crop(19, 462, 519)
    canvas.paste(_c19, (546, 1431), _c19)
except Exception:
    pass
layout["S86+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/20_icon_Hadestown.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (288, 2792), _c20)
except Exception:
    pass
layout["Hadestown"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/21_icon_New_York_NY.png
try:
    _c21 = get_crop(21, 390, 84)
    canvas.paste(_c21, (40, 120), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [40, 120, 430, 204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/22_icon_S86.png
try:
    _c22 = get_crop(22, 462, 519)
    canvas.paste(_c22, (546, 1431), _c22)
except Exception:
    pass
layout["S86+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/23_icon_May.png
try:
    _c23 = get_crop(23, 264, 183)
    canvas.paste(_c23, (1176, 2000), _c23)
except Exception:
    pass
layout["May"] = [1176, 2000, 1440, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/24_text_date.png
try:
    _c24 = get_crop(24, 114, 52)
    canvas.paste(_c24, (137, 208), _c24)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/25_text_Just_for_you.png
try:
    _c25 = get_crop(25, 306, 66)
    canvas.paste(_c25, (38, 1310), _c25)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/26_text_View_all.png
try:
    _c26 = get_crop(26, 264, 183)
    canvas.paste(_c26, (1176, 1248), _c26)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/27_text_Hadestown.png
try:
    _c27 = get_crop(27, 288, 168)
    canvas.paste(_c27, (288, 2792), _c27)
except Exception:
    pass
layout["Hadestown"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/28_clickable_Tracking.png
try:
    _c28 = get_crop(28, 72, 72)
    canvas.paste(_c28, (408, 1455), _c28)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/29_clickable_Tracking.png
try:
    _c29 = get_crop(29, 72, 72)
    canvas.paste(_c29, (906, 1455), _c29)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/30_clickable_Tickets.png
try:
    _c30 = get_crop(30, 288, 168)
    canvas.paste(_c30, (576, 2792), _c30)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_01_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-4/31_clickable_Account.png
try:
    _c31 = get_crop(31, 288, 168)
    canvas.paste(_c31, (1152, 2792), _c31)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]
