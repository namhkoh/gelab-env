# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_07
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9.png
# step_index: 7/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the mobile UI (PIL drawing)
# Uses provided variables: canvas (1440x2960 PIL Image) and draw (ImageDraw)
# and font_sm, font_md, font_lg, font_xl (not used)

# Overall page background (very light off-white)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 252))

# Status bar (top area) - muted grey strip
STATUS_H = 80
draw.rectangle((0, 0, 1440, STATUS_H), fill=(175, 175, 175))

# Header / toolbar background (behind search & controls)
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 240
draw.rectangle((0, HEADER_TOP, 1440, HEADER_BOTTOM), fill=(246, 247, 251))

# Subtle bottom divider under header
draw.line((48, HEADER_BOTTOM, 1392, HEADER_BOTTOM), fill=(220, 220, 225), width=2)

# Secondary divider line beneath filter area (approximate)
FILTER_DIV_Y = 420
draw.line((48, FILTER_DIV_Y, 1392, FILTER_DIV_Y), fill=(235, 235, 240), width=1)

# Large card background (first event card container) with soft shadow
card1_outer = (36, 652, 1404, 1780)
shadow_offset = (8, 10)
# Card shadow
draw.rounded_rectangle(
    (card1_outer[0] + shadow_offset[0], card1_outer[1] + shadow_offset[1],
     card1_outer[2] + shadow_offset[0], card1_outer[3] + shadow_offset[1]),
    radius=28, fill=(238, 238, 242)
)
# Card body (white)
draw.rounded_rectangle(card1_outer, radius=28, fill=(255, 255, 255))

# Image/background area inside first card (underneath the image that will be pasted)
# Use a muted deep-blue to approximate the water behind the cruise image
img1_bbox = (48, 676, 1392, 1160)  # top portion reserved for the image
draw.rounded_rectangle(img1_bbox, radius=18, fill=(28, 92, 120))

# Subtle separator between image and textual area inside card (thin line)
draw.line((60, img1_bbox[3] + 12, 1380, img1_bbox[3] + 12), fill=(245, 245, 247), width=1)

# Second card (Puppy Yoga) container with shadow
card2_outer = (36, 1776, 1404, 2848)
# Card shadow
draw.rounded_rectangle(
    (card2_outer[0] + shadow_offset[0], card2_outer[1] + shadow_offset[1],
     card2_outer[2] + shadow_offset[0], card2_outer[3] + shadow_offset[1]),
    radius=28, fill=(238, 238, 242)
)
# Card body
draw.rounded_rectangle(card2_outer, radius=28, fill=(255, 255, 255))

# Image/background area inside second card (underneath the yoga image)
# Use a warm muted red/pink hint under the photo
img2_bbox = (48, 1815, 1392, 2608)
draw.rounded_rectangle(img2_bbox, radius=18, fill=(220, 90, 90))

# Divider line above bottom navigation
NAV_TOP = 2848
draw.line((48, NAV_TOP, 1392, NAV_TOP), fill=(225, 225, 230), width=2)

# Bottom navigation bar background (keeps icons area free for paste)
draw.rectangle((0, NAV_TOP, 1440, 2960), fill=(255, 255, 255))
# Tiny top shadow for nav bar
draw.rectangle((0, NAV_TOP, 1440, NAV_TOP + 3), fill=(240, 240, 245))

# Additional subtle full-width separators to cadence the feed
draw.line((36, 1320, 1404, 1320), fill=(248, 248, 249), width=1)
draw.line((36, 2240, 1404, 2240), fill=(248, 248, 249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/04_icon_Foo.png
try:
    _c4 = get_crop(4, 151, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1433, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2331), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2331), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/09_icon_8.07.png
try:
    _c9 = get_crop(9, 122, 111)
    canvas.paste(_c9, (57, 117), _c9)
except Exception:
    pass
layout["8.07"] = [57, 117, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/10_icon_ALL_White_Attire_200Os_Throwbacks_Cruise.png
try:
    _c10 = get_crop(10, 1344, 1091)
    canvas.paste(_c10, (48, 676), _c10)
except Exception:
    pass
layout["ALL_White_Attire_200Os_Th"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 65)
    canvas.paste(_c11, (1152, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1152, 0, 1205, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/12_icon_ALL_White_Attire_200Os_Throwbacks_Cruise.png
try:
    _c12 = get_crop(12, 1344, 1091)
    canvas.paste(_c12, (48, 676), _c12)
except Exception:
    pass
layout["ALL_White_Attire_200Os_Th"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/13_icon_8.07.png
try:
    _c13 = get_crop(13, 61, 64)
    canvas.paste(_c13, (180, 0), _c13)
except Exception:
    pass
layout["8.07"] = [180, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 92, 62)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1304, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 69, 63)
    canvas.paste(_c15, (307, 0), _c15)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/16_icon_8.07.png
try:
    _c16 = get_crop(16, 62, 65)
    canvas.paste(_c16, (113, 0), _c16)
except Exception:
    pass
layout["8.07"] = [113, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 54, 64)
    canvas.paste(_c17, (246, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/19_icon_Puppy_Yoga_Stretch_Relax_and_Bring_your.png
try:
    _c19 = get_crop(19, 1344, 1001)
    canvas.paste(_c19, (48, 1815), _c19)
except Exception:
    pass
layout["Puppy_Yoga:_Stretch,_Rela"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 59, 59)
    canvas.paste(_c20, (1317, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1317, 0, 1376, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/21_icon_Chicago.png
try:
    _c21 = get_crop(21, 417, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/22_icon_10_00_AM_CDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["10:00_AM_CDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/23_icon_Promoted.png
try:
    _c23 = get_crop(23, 261, 68)
    canvas.paste(_c23, (68, 1658), _c23)
except Exception:
    pass
layout["Promoted"] = [68, 1658, 329, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/24_icon_21c_Museum_Hotel_Chicago_East_Ontario_St.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["21c_Museum_Hotel_Chicago,"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/25_icon_Search_forae.png
try:
    _c25 = get_crop(25, 52, 61)
    canvas.paste(_c25, (383, 2), _c25)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/26_icon_Anita_Dee_Yacht_Charters.png
try:
    _c26 = get_crop(26, 493, 65)
    canvas.paste(_c26, (68, 1592), _c26)
except Exception:
    pass
layout["Anita_Dee_Yacht_Charters"] = [68, 1592, 561, 1657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/27_icon_21c_Museum_Hotel_Chicago_East_Ontario_St.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["21c_Museum_Hotel_Chicago,"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/28_icon_21c_Museum_Hotel_Chicago_East_Ontario_St.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (576, 2804), _c28)
except Exception:
    pass
layout["21c_Museum_Hotel_Chicago,"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/29_icon_Sun_May_12.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Sun,_May_12"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 40, 61)
    canvas.paste(_c30, (1274, 0), _c30)
except Exception:
    pass
layout["icon_30"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/31_icon_Anita_Dee_Yacht_Charters.png
try:
    _c31 = get_crop(31, 46, 62)
    canvas.paste(_c31, (284, 1661), _c31)
except Exception:
    pass
layout["Anita_Dee_Yacht_Charters"] = [284, 1661, 330, 1723]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/32_icon_8.07.png
try:
    _c32 = get_crop(32, 152, 64)
    canvas.paste(_c32, (7, 0), _c32)
except Exception:
    pass
layout["8.07"] = [7, 0, 159, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_07_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-9/33_text_10_000_events.png
try:
    _c33 = get_crop(33, 359, 103)
    canvas.paste(_c33, (54, 410), _c33)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]
