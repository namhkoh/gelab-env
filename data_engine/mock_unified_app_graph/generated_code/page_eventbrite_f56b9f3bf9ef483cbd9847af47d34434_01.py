# page_id: page_eventbrite_f56b9f3bf9ef483cbd9847af47d34434_01
# screenshot: 2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3.png
# step_index: 1/8
# task: Open Eventbrite. Look up "Gardening" events. Filter by events happening this week. Select the first event from the results. Follow the organizer and where is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar (top ~50px) - light gray background to match screenshot status area
STATUS_H = 80
draw.rectangle((0, 0, 1440, STATUS_H), fill="#D0D0D0")

# Header area (behind search field) - keep white but add subtle bottom divider/shadow
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 220
draw.rectangle((0, HEADER_TOP, 1440, HEADER_BOTTOM), fill="#FFFFFF")
draw.line((24, HEADER_BOTTOM, 1440-24, HEADER_BOTTOM), fill="#E6E6E9", width=1)

# Large content area remains white; add top-level subtle divider under the main title area
TITLE_DIV_Y = 320
draw.line((48, TITLE_DIV_Y, 1440-48, TITLE_DIV_Y), fill="#F0F1F4", width=1)

# Card backgrounds (rounded rectangles) for each content row using the detected group bounds.
# We draw soft drop shadows then white rounded cards with light borders.
card_specs = [
    (48, 490, 48 + 1344, 490 + 396),
    (48, 886, 48 + 1344, 886 + 396),
    (48, 1282, 48 + 1344, 1282 + 396),
    (48, 1678, 48 + 1344, 1678 + 396),
    (48, 2074, 48 + 1344, 2074 + 396),
    (48, 2470, 48 + 1344, 2470 + 346),
]
radius = 14
for (x1, y1, x2, y2) in card_specs:
    # shadow
    shadow_box = (x1 + 6, y1 + 8, x2 + 6, y2 + 8)
    draw.rounded_rectangle(shadow_box, radius=radius, fill="#EEEEF0")
    # card
    draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill="#FFFFFF", outline="#F3F4F6", width=1)
    # top separator subtle line for card
    draw.line((x1 + 12, y1, x2 - 12, y1), fill="#F5F6F8", width=1)

# Thin separators between content rows (full-bleed subtle lines)
separator_ys = [y2 + 12 for (_, _, _, y2) in card_specs[:-1]]
for sy in separator_ys:
    draw.line((48, sy, 1440-48, sy), fill="#F0F1F4", width=1)

# Bottom navigation bar background and top divider (space reserved for clickable nav elements)
NAV_TOP = 2804
draw.rectangle((0, NAV_TOP, 1440, 2960), fill="#FFFFFF")
draw.line((24, NAV_TOP, 1440-24, NAV_TOP), fill="#E6E6E9", width=1)
# subtle nav shadow above the nav bar
draw.line((24, NAV_TOP+1, 1440-24, NAV_TOP+1), fill="#F6F7F9", width=1)

# Small edge gutters / safe area subtle guidelines (not visible in final UI but help structure)
# (Very faint lines on left/right edges)
draw.line((24, 0, 24, 2960), fill="#FFFFFF", width=1)
draw.line((1440-24, 0, 1440-24, 2960), fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/00_icon_Chicago.png
try:
    _c0 = get_crop(0, 388, 117)
    canvas.paste(_c0, (526, 2651), _c0)
except Exception:
    pass
layout["Chicago"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/01_icon_CyPo6.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["CyPo6"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/02_icon_ripg_-_LeaTG_Atans.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2074), _c2)
except Exception:
    pass
layout["ripg_-_LeaTG_Atans"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/03_icon_Okstore.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["Okstore"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/04_icon_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/05_icon_Sat_Oct_19.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 490), _c5)
except Exception:
    pass
layout["Sat,_Oct_19"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/06_icon_Dovetail_Brewery.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Dovetail_Brewery"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 2347), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 1935), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1284, 2347), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1140, 1143), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/12_icon_7940_Wolcott_Ave_apt_2_Chicago_IL_USA.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 490), _c12)
except Exception:
    pass
layout["7940_$_Wolcott_Ave_apt_2,"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/13_icon_Favorite_button.png
try:
    _c13 = get_crop(13, 144, 125)
    canvas.paste(_c13, (1140, 761), _c13)
except Exception:
    pass
layout["Favorite_button"] = [1140, 761, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 125)
    canvas.paste(_c14, (1284, 761), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 761, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/15_icon_Joliet.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["Joliet"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/17_icon_5.09.png
try:
    _c17 = get_crop(17, 107, 101)
    canvas.paste(_c17, (39, 121), _c17)
except Exception:
    pass
layout["5.09"] = [39, 121, 146, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/18_icon_through_thc_chi.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (0, 2804), _c18)
except Exception:
    pass
layout["through_thc_chi"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/19_icon_Overflow_menu_button.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1284, 1143), _c19)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/20_icon_ON.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["ON"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 60, 58)
    canvas.paste(_c21, (312, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/22_icon_49_creator_followers.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 886), _c22)
except Exception:
    pass
layout["49_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/23_icon_5.09.png
try:
    _c23 = get_crop(23, 56, 60)
    canvas.paste(_c23, (182, 2), _c23)
except Exception:
    pass
layout["5.09"] = [182, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 50, 59)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/25_icon_Indie_Bookstore_Day_at_Goblin_Market.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1282), _c25)
except Exception:
    pass
layout["Indie_Bookstore_Day_at_Go"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/26_icon_Planting_Seeds_bilingual.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 2074), _c26)
except Exception:
    pass
layout["Planting_Seeds_(bilingual"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 48, 53)
    canvas.paste(_c27, (1321, 7), _c27)
except Exception:
    pass
layout["icon_27"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 58, 58)
    canvas.paste(_c28, (1212, 4), _c28)
except Exception:
    pass
layout["icon_28"] = [1212, 4, 1270, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/29_icon_5.09.png
try:
    _c29 = get_crop(29, 58, 61)
    canvas.paste(_c29, (115, 2), _c29)
except Exception:
    pass
layout["5.09"] = [115, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 44, 56)
    canvas.paste(_c30, (385, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 41, 55)
    canvas.paste(_c31, (1272, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/32_icon_73_creator_followers.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["73_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/33_icon_Self-Love_in_Nature_Releasing_Grief_thro.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Self-Love_in_Nature:_Rele"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/34_icon_Grief_R.png
try:
    _c34 = get_crop(34, 1344, 346)
    canvas.paste(_c34, (48, 2470), _c34)
except Exception:
    pass
layout["Grief_R"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/35_icon_6_00_PM_CDT.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/36_icon_Dovetail_Brewery.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1678), _c36)
except Exception:
    pass
layout["Dovetail_Brewery"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/37_icon_Discover_Your_Path_To_Healing_With_Our_G.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["Discover_Your_Path_To_Hea"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/38_text_5.09.png
try:
    _c38 = get_crop(38, 91, 45)
    canvas.paste(_c38, (20, 15), _c38)
except Exception:
    pass
layout["5.09"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/40_text_Tue_May_7.png
try:
    _c40 = get_crop(40, 191, 43)
    canvas.paste(_c40, (390, 2525), _c40)
except Exception:
    pass
layout["Tue,_May_7"] = [390, 2525, 581, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/41_text_6_00_PM_CDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/42_text_Joliet.png
try:
    _c42 = get_crop(42, 96, 38)
    canvas.paste(_c42, (390, 2723), _c42)
except Exception:
    pass
layout["Joliet"] = [390, 2723, 486, 2761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/43_clickable_Favorites.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (576, 2804), _c43)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/44_clickable_Tickets.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (864, 2804), _c44)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_01_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
