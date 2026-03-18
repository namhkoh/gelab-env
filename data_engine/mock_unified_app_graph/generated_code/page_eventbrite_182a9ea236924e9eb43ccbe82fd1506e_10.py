# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_10
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12.png
# step_index: 10/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background
draw.rectangle([(0, 0), (1440, 2960)], fill=(246, 247, 249))  # subtle off-white background

# Status bar area (top ~56px)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill=(189, 189, 189))  # light gray status bar

# Header / Search area background (under status bar)
header_top = status_h
header_bottom = 200
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Search field background (centered with side margins)
search_x0, search_x1 = 48, 1392
search_y0, search_y1 = header_top + 16, header_top + 96
draw.rounded_rectangle([(search_x0, search_y0), (search_x1, search_y1)], radius=14, fill=(249, 250, 251), outline=None)

# Thin divider below header
draw.line([(48, header_bottom), (1392, header_bottom)], fill=(230, 231, 235), width=2)

# Separator line a bit lower (between chips area and list)
draw.line([(48, 260), (1392, 260)], fill=(240, 241, 243), width=1)

# First content card background (behind the title block at ~y=525)
card1_x0, card1_x1 = 48, 1392
card1_y0, card1_y1 = 520, 1030
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)], radius=14, fill=(255, 255, 255), outline=(235, 236, 239))

# Subtle top shadow for card1 (simulated by a thin darker band)
draw.line([(card1_x0 + 6, card1_y0 + 2), (card1_x1 - 6, card1_y0 + 2)], fill=(242, 243, 244), width=2)

# Divider between first card and the following list area
draw.line([(48, card1_y1 + 12), (1392, card1_y1 + 12)], fill=(235, 236, 239), width=1)

# Large section card background for the block starting around y=1076 (covers titles and content)
card2_x0, card2_x1 = 48, 1392
card2_y0, card2_y1 = 1076, 2184  # matches detected block height
draw.rounded_rectangle([(card2_x0, card2_y0), (card2_x1, card2_y1)], radius=16, fill=(255, 255, 255), outline=(235, 236, 239))

# Small badge background placeholder area (not drawing text/icons) - a soft rounded rectangle where badges will appear
badge_x0, badge_x1 = 60, 200
badge_y0, badge_y1 = 1748, 1792  # approximate area for small badges like "Free"
draw.rounded_rectangle([(badge_x0, badge_y0), (badge_x1, badge_y1)], radius=8, fill=(241, 249, 241), outline=(220, 235, 220))

# Rounded background behind the large event image area (so pasted image sits on a card)
image_card_x0, image_card_x1 = 48, 1392
image_card_y0, image_card_y1 = 2232, 2232 + 584
draw.rounded_rectangle([(image_card_x0, image_card_y0), (image_card_x1, image_card_y1)], radius=20, fill=(244, 245, 246), outline=(232, 233, 235))

# Thin separators between major content sections
draw.line([(48, 1038), (1392, 1038)], fill=(240, 241, 243), width=1)
draw.line([(48, 2188), (1392, 2188)], fill=(240, 241, 243), width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
draw.line([(0, nav_top), (1440, nav_top)], fill=(232, 233, 235), width=2)

# Selected indicator for the center navigation item (subtle orange dot)
selected_center_x = 288 + 144  # center of the second nav slot (index at x=288)
indicator_y = nav_top + 22
draw.ellipse([(selected_center_x - 8, indicator_y - 8), (selected_center_x + 8, indicator_y + 8)], fill=(236, 97, 46))

# Soft top shadow above the entire content area (under header)
draw.line([(0, header_bottom + 2), (1440, header_bottom + 2)], fill=(245, 246, 247), width=4)

# Final subtle global vignette lines to mimic card separations (very light)
for y in (320, 640, 920, 1280, 1600, 1920, 2240):
    draw.line([(48, y), (1392, y)], fill=(250, 250, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/00_icon_Music.png
try:
    _c0 = get_crop(0, 196, 111)
    canvas.paste(_c0, (875, 406), _c0)
except Exception:
    pass
layout["Music"] = [875, 406, 1071, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/01_icon_Tomorrow.png
try:
    _c1 = get_crop(1, 1344, 191)
    canvas.paste(_c1, (48, 72), _c1)
except Exception:
    pass
layout["Tomorrow"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/02_icon_Business.png
try:
    _c2 = get_crop(2, 250, 111)
    canvas.paste(_c2, (1074, 406), _c2)
except Exception:
    pass
layout["Business"] = [1074, 406, 1324, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 434, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/04_icon_Business.png
try:
    _c4 = get_crop(4, 100, 110)
    canvas.paste(_c4, (1328, 407), _c4)
except Exception:
    pass
layout["Business"] = [1328, 407, 1428, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/05_icon_Going_fast.png
try:
    _c5 = get_crop(5, 272, 86)
    canvas.paste(_c5, (90, 526), _c5)
except Exception:
    pass
layout["Going_fast"] = [90, 526, 362, 612]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/06_icon_New_York.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1592), _c6)
except Exception:
    pass
layout["New_York"] = [1092, 1592, 1236, 1736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/07_icon_New_York.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1592), _c7)
except Exception:
    pass
layout["New_York"] = [1236, 1592, 1380, 1736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/08_icon_NEW_YORK.png
try:
    _c8 = get_crop(8, 1344, 1108)
    canvas.paste(_c8, (48, 1076), _c8)
except Exception:
    pass
layout["NEW_YORK"] = [48, 1076, 1392, 2184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 62)
    canvas.paste(_c9, (247, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 1, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/10_icon_9.32.png
try:
    _c10 = get_crop(10, 124, 115)
    canvas.paste(_c10, (56, 114), _c10)
except Exception:
    pass
layout["9.32"] = [56, 114, 180, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/11_icon_9.32.png
try:
    _c11 = get_crop(11, 58, 63)
    canvas.paste(_c11, (181, 0), _c11)
except Exception:
    pass
layout["9.32"] = [181, 0, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 93, 61)
    canvas.paste(_c12, (1208, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1208, 0, 1301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 60, 61)
    canvas.paste(_c13, (1316, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1316, 0, 1376, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/14_icon_Search_forae.png
try:
    _c14 = get_crop(14, 59, 63)
    canvas.paste(_c14, (312, 1), _c14)
except Exception:
    pass
layout["Search_forae"] = [312, 1, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/15_icon_9.32.png
try:
    _c15 = get_crop(15, 58, 65)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["9.32"] = [114, 0, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/16_icon_WBONY_TheRealReal_Sustainability_in.png
try:
    _c16 = get_crop(16, 1344, 503)
    canvas.paste(_c16, (48, 525), _c16)
except Exception:
    pass
layout["WBONY_&_TheRealReal_Susta"] = [48, 525, 1392, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/17_icon_IO.O0AM_EDT.png
try:
    _c17 = get_crop(17, 1344, 1108)
    canvas.paste(_c17, (48, 1076), _c17)
except Exception:
    pass
layout["IO.O0AM_EDT"] = [48, 1076, 1392, 2184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/18_icon_New_York.png
try:
    _c18 = get_crop(18, 434, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/20_icon_Search_events.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 48, 63)
    canvas.paste(_c21, (383, 1), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/22_icon_Tickets.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/23_icon_More.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/24_icon_Favorites.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/25_icon_Home.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 39, 62)
    canvas.paste(_c26, (1275, 0), _c26)
except Exception:
    pass
layout["icon_26"] = [1275, 0, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/27_icon_Free.png
try:
    _c27 = get_crop(27, 126, 75)
    canvas.paste(_c27, (91, 1770), _c27)
except Exception:
    pass
layout["Free"] = [91, 1770, 217, 1845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/28_icon_Thu_Mar_21.png
try:
    _c28 = get_crop(28, 506, 53)
    canvas.paste(_c28, (88, 792), _c28)
except Exception:
    pass
layout["Thu,_Mar_21"] = [88, 792, 594, 845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/29_text_9.32.png
try:
    _c29 = get_crop(29, 96, 49)
    canvas.paste(_c29, (16, 12), _c29)
except Exception:
    pass
layout["9.32"] = [16, 12, 112, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/30_text_Arrive.png
try:
    _c30 = get_crop(30, 122, 50)
    canvas.paste(_c30, (91, 864), _c30)
except Exception:
    pass
layout["Arrive"] = [91, 864, 213, 914]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/31_text_Promoted.png
try:
    _c31 = get_crop(31, 195, 45)
    canvas.paste(_c31, (94, 933), _c31)
except Exception:
    pass
layout["Promoted"] = [94, 933, 289, 978]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_10_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-12/32_clickable_Event_s_image.png
try:
    _c32 = get_crop(32, 1344, 584)
    canvas.paste(_c32, (48, 2232), _c32)
except Exception:
    pass
layout["Event's_image"] = [48, 2232, 1392, 2816]
