# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_01
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4.png
# step_index: 1/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (dominant off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")

# Top status bar (approx ~72px height)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#ECECEC")

# Header area beneath status bar
header_top = status_h
header_bottom = 240
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# subtle divider / shadow below header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#E6E6E6", width=1)

# NOTE: avoid drawing inside the large hero image area (detected element at (48,360) size 1344x840)
hero_box = (48, 360, 48+1344, 360+840)

# "Just for you" cards background container (rounded rectangle).
# Place it below the hero area, but make sure to not paint over the hero_box.
jf_top = hero_box[3] + 40  # start some space below hero
jf_bottom = jf_top + 420
jf_left = 24
jf_right = 1440 - 24
draw.rounded_rectangle([(jf_left, jf_top), (jf_right, jf_bottom)],
                       radius=20, fill="#FFFFFF", outline="#EDEDED", width=1)

# subtle inner shadow for the "Just for you" container
draw.line([(jf_left+4, jf_bottom), (jf_right-4, jf_bottom)], fill="#F2F2F2", width=2)

# Divider between "Just for you" and next section
sep_y = jf_bottom + 36
draw.line([(24, sep_y), (1440-24, sep_y)], fill="#EDEDED", width=1)

# Trending events section background block (light)
tr_top = sep_y + 28
tr_left = 24
tr_right = 1440 - 24
# Create a subtle white panel for the list
draw.rectangle([(tr_left, tr_top), (tr_right, tr_top + 680)], fill="#FFFFFF", outline="#EDEDED")

# Draw three list item separators (approx positions matching screenshot spacing)
item_h = 80
for i in range(4):
    y = tr_top + 20 + i * (item_h + 24)
    # item background (keep it white, show separators)
    draw.rectangle([(tr_left + 12, y), (tr_right - 12, y + item_h)], fill="#FFFFFF")
    # bottom separator line
    draw.line([(tr_left + 12, y + item_h + 6), (tr_right - 12, y + item_h + 6)], fill="#EFEFEF", width=1)

# Big horizontal separator above bottom navigation
nav_top = 2760
draw.line([(24, nav_top), (1440-24, nav_top)], fill="#E6E6E6", width=1)

# Bottom navigation bar background (slightly elevated white)
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")

# Rounded top corners for nav area (simulate subtle curve)
draw.pieslice([(-20, nav_top-40), (40, nav_top+40)], 90, 180, fill="#FFFFFF")
draw.pieslice([(1440-40, nav_top-40), (1440+20, nav_top+40)], 270, 360, fill="#FFFFFF")

# Light shadow above nav bar for separation
draw.line([(24, nav_top+4), (1440-24, nav_top+4)], fill="#F2F2F2", width=2)

# Small decorative vertical separators in the trending panel to structure columns (non-icon background only)
for x in (tr_left + 220, tr_left + 640, tr_left + 980):
    draw.line([(x, tr_top + 8), (x, tr_top + 8 + 680)], fill="#FAFAFA", width=1)

# Top-left corner decorative rounded rectangle (subtle card hint behind location text)
# Keep it minimal and ensure not to draw over icons/text areas precisely
loc_box = (24, header_top + 20, 420, header_top + 120)
draw.rounded_rectangle([loc_box[0:2], loc_box[2:4]], radius=12, fill="#FFFFFF", outline="#F0F0F0")

# Right-side filter icon background hint (header area), small rounded rect (background only)
filter_box = (1390-60, header_top + 28, 1390-8, header_top + 100)
draw.rounded_rectangle([filter_box[0:2], filter_box[2:4]], radius=12, fill="#FFFFFF", outline="#F0F0F0")

# Final subtle global vignette border (very light) to mimic app window edges
draw.rectangle([(0, 0), (1440, 8)], fill="#F6F6F6")
draw.rectangle([(0, 2960-8), (1440, 2960)], fill="#F6F6F6")
draw.rectangle([(0, 0), (8, 2960)], fill="#F6F6F6")
draw.rectangle([(1440-8, 0), (1440, 2960)], fill="#F6F6F6")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/00_icon_St._James_Theatre.png
try:
    _c0 = get_crop(0, 1309, 236)
    canvas.paste(_c0, (0, 2183), _c0)
except Exception:
    pass
layout["St._James_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/01_icon_S87.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (546, 1431), _c1)
except Exception:
    pass
layout["S87+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 101, 150)
    canvas.paste(_c2, (1339, 2464), _c2)
except Exception:
    pass
layout["icon_2"] = [1339, 2464, 1440, 2614]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/03_icon_View_all.png
try:
    _c3 = get_crop(3, 99, 148)
    canvas.paste(_c3, (1341, 2228), _c3)
except Exception:
    pass
layout["View_all"] = [1341, 2228, 1440, 2376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/04_icon_840.png
try:
    _c4 = get_crop(4, 144, 240)
    canvas.paste(_c4, (1260, 72), _c4)
except Exception:
    pass
layout["840"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/05_icon_8.27_my.png
try:
    _c5 = get_crop(5, 53, 58)
    canvas.paste(_c5, (115, 3), _c5)
except Exception:
    pass
layout["8.27_my"] = [115, 3, 168, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/06_icon_840.png
try:
    _c6 = get_crop(6, 98, 63)
    canvas.paste(_c6, (1216, 1), _c6)
except Exception:
    pass
layout["840"] = [1216, 1, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/07_icon_8.27_my.png
try:
    _c7 = get_crop(7, 56, 60)
    canvas.paste(_c7, (182, 2), _c7)
except Exception:
    pass
layout["8.27_my"] = [182, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/08_icon_NCAA_M_Basketball_Brooklyn.png
try:
    _c8 = get_crop(8, 1309, 236)
    canvas.paste(_c8, (0, 2419), _c8)
except Exception:
    pass
layout["NCAA_M_Basketball_Brookly"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 56)
    canvas.paste(_c9, (316, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [316, 5, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 62)
    canvas.paste(_c10, (1320, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1320, 2, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/11_icon_BARCLAYS_CEME.png
try:
    _c11 = get_crop(11, 462, 519)
    canvas.paste(_c11, (48, 1431), _c11)
except Exception:
    pass
layout["BARCLAYS_CEME"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/12_icon_New_York_NY.png
try:
    _c12 = get_crop(12, 52, 57)
    canvas.paste(_c12, (247, 5), _c12)
except Exception:
    pass
layout["New_York,_NY"] = [247, 5, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 45, 64)
    canvas.paste(_c13, (1155, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1155, 1, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 98, 113)
    canvas.paste(_c14, (1342, 2698), _c14)
except Exception:
    pass
layout["icon_14"] = [1342, 2698, 1440, 2811]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/15_icon_Hadestown.png
try:
    _c15 = get_crop(15, 288, 162)
    canvas.paste(_c15, (0, 2792), _c15)
except Exception:
    pass
layout["Hadestown"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/16_icon_Tracking.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (864, 2792), _c16)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/17_icon_S2_4_D.png
try:
    _c17 = get_crop(17, 125, 146)
    canvas.paste(_c17, (1135, 2471), _c17)
except Exception:
    pass
layout["S2_(#4_D="] = [1135, 2471, 1260, 2617]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/18_icon_TIcKETS.png
try:
    _c18 = get_crop(18, 1344, 840)
    canvas.paste(_c18, (48, 360), _c18)
except Exception:
    pass
layout["TIcKETS"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/19_icon_Hadestown.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (288, 2792), _c19)
except Exception:
    pass
layout["Hadestown"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/20_icon_S87.png
try:
    _c20 = get_crop(20, 462, 519)
    canvas.paste(_c20, (546, 1431), _c20)
except Exception:
    pass
layout["S87+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/21_icon_New_York_NY.png
try:
    _c21 = get_crop(21, 391, 84)
    canvas.paste(_c21, (39, 120), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [39, 120, 430, 204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/22_icon_Account.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (1152, 2792), _c22)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/23_icon_Mar_22_._Barclays_Center.png
try:
    _c23 = get_crop(23, 288, 168)
    canvas.paste(_c23, (576, 2792), _c23)
except Exception:
    pass
layout["Mar_22_._Barclays_Center"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/24_text_date.png
try:
    _c24 = get_crop(24, 114, 52)
    canvas.paste(_c24, (137, 208), _c24)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/25_text_Just_for_you.png
try:
    _c25 = get_crop(25, 306, 66)
    canvas.paste(_c25, (38, 1310), _c25)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/26_text_View_all.png
try:
    _c26 = get_crop(26, 264, 183)
    canvas.paste(_c26, (1176, 1248), _c26)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/27_text_Trending_events.png
try:
    _c27 = get_crop(27, 423, 81)
    canvas.paste(_c27, (38, 2054), _c27)
except Exception:
    pass
layout["Trending_events"] = [38, 2054, 461, 2135]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/28_text_View_all.png
try:
    _c28 = get_crop(28, 264, 183)
    canvas.paste(_c28, (1176, 2000), _c28)
except Exception:
    pass
layout["View_all"] = [1176, 2000, 1440, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/29_text_Hadestown.png
try:
    _c29 = get_crop(29, 288, 168)
    canvas.paste(_c29, (288, 2792), _c29)
except Exception:
    pass
layout["Hadestown"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/30_clickable_Tracking.png
try:
    _c30 = get_crop(30, 72, 72)
    canvas.paste(_c30, (408, 1455), _c30)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_01_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (906, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
