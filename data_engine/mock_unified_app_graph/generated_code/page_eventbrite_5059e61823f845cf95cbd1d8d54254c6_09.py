# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_09
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11.png
# step_index: 9/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# fill overall background (dominant color: white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar (top ~56px) - neutral grey background matching screenshot
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill="#9E9E9E")

# Header area (below status bar) — keep white but draw a bold blue underline
header_top = status_h
header_bottom = 144
# subtle background (white, same as canvas) to ensure clean area
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
# blue underline under the header
underline_y = header_bottom - 4
draw.rectangle((48, underline_y, 1392, underline_y + 4), fill="#2F5CE6")

# subtle 1px divider just below the underline to emulate slight separation
draw.rectangle((48, underline_y + 4, 1392, underline_y + 5), fill="#E6E6E6")

# Section separators / subtle horizontal rules for content flow
# A divider under the "Popular" list region
popular_div_y = 300
draw.rectangle((32, popular_div_y, 1408, popular_div_y + 1), fill="#E9E9EF")

# Card-like rounded backgrounds for each event row (do NOT draw any icons/text)
# Use slightly off-white cards with soft corners and a faint border to sit behind pasted content.
event_rows_centers = [1117, 1513, 1909, 2305]  # approximate y positions from detected elements
card_left = 32
card_right = 1408
card_height = 220
card_radius = 12
card_fill = "#FBFBFD"
card_border = "#F0F0F3"
for cy in event_rows_centers:
    top = cy - 40
    bottom = top + card_height
    # keep cards within canvas
    if top < header_bottom + 20:
        top = header_bottom + 20
        bottom = top + card_height
    if bottom > 2800:
        bottom = 2800
    draw.rounded_rectangle((card_left, top, card_right, bottom), radius=card_radius, fill=card_fill, outline=card_border, width=1)
    # subtle separator line under each card
    sep_y = bottom + 12
    if sep_y < 2800:
        draw.rectangle((48, sep_y, 1392, sep_y + 1), fill="#E6E6E6")

# Bottom navigation bar background (approx 156px high) with top border
nav_top = 2804
nav_bottom = 2960
draw.rectangle((0, nav_top, 1440, nav_bottom), fill="#FFFFFF")
draw.rectangle((0, nav_top, 1440, nav_top + 1), fill="#E6E6E6")

# Active nav indicator (orange) behind the second nav slot center
# Nav slots are 288px wide (0,288,576,864,1152), center X for second slot = 288 + 144 = 432
active_center_x = 288 + 144
indicator_width = 64
indicator_height = 6
ind_left = active_center_x - indicator_width // 2
ind_right = active_center_x + indicator_width // 2
ind_top = nav_top + 14
ind_bottom = ind_top + indicator_height
draw.rounded_rectangle((ind_left, ind_top, ind_right, ind_bottom), radius=6, fill="#F05A28")

# Lightweight left gutter vertical guide (visual alignment aid, very subtle)
draw.rectangle((24, header_bottom + 10, 28, nav_top - 20), fill="#FFFFFF")

# A faint drop shadow beneath header area for depth
shadow_top = header_bottom
shadow_bottom = header_bottom + 6
draw.rectangle((0, shadow_top, 1440, shadow_bottom), fill="#F2F2F4")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/00_icon_Men.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2305), _c0)
except Exception:
    pass
layout["Men"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/01_icon_Events.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1117), _c1)
except Exception:
    pass
layout["Events"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/02_icon_CPR_4AII.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1909), _c2)
except Exception:
    pass
layout["CPR_4AII"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/03_icon_8_257_creator_followers.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1909), _c3)
except Exception:
    pass
layout["8_257_creator_followers"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/04_icon_Education.png
try:
    _c4 = get_crop(4, 56, 59)
    canvas.paste(_c4, (313, 4), _c4)
except Exception:
    pass
layout["Education"] = [313, 4, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/05_icon_9_00_AM_PDT.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1513), _c5)
except Exception:
    pass
layout["9:00_AM_PDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/06_icon_7.35.png
try:
    _c6 = get_crop(6, 53, 60)
    canvas.paste(_c6, (183, 3), _c6)
except Exception:
    pass
layout["7.35"] = [183, 3, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/07_icon_Sat.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1513), _c7)
except Exception:
    pass
layout["Sat,"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/08_icon_7.35.png
try:
    _c8 = get_crop(8, 57, 61)
    canvas.paste(_c8, (115, 3), _c8)
except Exception:
    pass
layout["7.35"] = [115, 3, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 55)
    canvas.paste(_c9, (253, 6), _c9)
except Exception:
    pass
layout["icon_9"] = [253, 6, 296, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 43, 69)
    canvas.paste(_c10, (1157, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1157, 0, 1200, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (864, 2804), _c11)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/12_icon_7.35.png
try:
    _c12 = get_crop(12, 113, 101)
    canvas.paste(_c12, (60, 120), _c12)
except Exception:
    pass
layout["7.35"] = [60, 120, 173, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/13_icon_Educational_professionals.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 1117), _c13)
except Exception:
    pass
layout["Educational_professionals"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/14_icon_Sun_Apr_28.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (288, 2804), _c14)
except Exception:
    pass
layout["Sun,_Apr_28"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 93, 66)
    canvas.paste(_c15, (1217, 0), _c15)
except Exception:
    pass
layout["Cancel"] = [1217, 0, 1310, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/16_icon_II_O0_AM_PDT.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (576, 2804), _c16)
except Exception:
    pass
layout["II:O0_AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/17_icon_Education.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 50, 66)
    canvas.paste(_c18, (1320, 0), _c18)
except Exception:
    pass
layout["Cancel"] = [1320, 0, 1370, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1099, 96), _c19)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/20_icon_City_Club_LA.png
try:
    _c20 = get_crop(20, 201, 52)
    canvas.paste(_c20, (390, 2571), _c20)
except Exception:
    pass
layout["City_Club_LA"] = [390, 2571, 591, 2623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/21_icon_7.35.png
try:
    _c21 = get_crop(21, 91, 60)
    canvas.paste(_c21, (16, 3), _c21)
except Exception:
    pass
layout["7.35"] = [16, 3, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 94, 99)
    canvas.paste(_c22, (33, 527), _c22)
except Exception:
    pass
layout["icon_22"] = [33, 527, 127, 626]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 149, 144)
    canvas.paste(_c23, (1243, 97), _c23)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/24_icon_Education.png
try:
    _c24 = get_crop(24, 47, 61)
    canvas.paste(_c24, (384, 3), _c24)
except Exception:
    pass
layout["Education"] = [384, 3, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 84, 93)
    canvas.paste(_c25, (40, 768), _c25)
except Exception:
    pass
layout["icon_25"] = [40, 768, 124, 861]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/26_icon_Ist_Nature-Based_Education_Summit.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1513), _c26)
except Exception:
    pass
layout["Ist_Nature-Based_Educatio"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 87, 94)
    canvas.paste(_c27, (37, 647), _c27)
except Exception:
    pass
layout["icon_27"] = [37, 647, 124, 741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/28_icon_7_00_PM_GMT_O1_O0.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1117), _c28)
except Exception:
    pass
layout["7:00_PM_GMT+O1:O0"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/30_icon_financial_education.png
try:
    _c30 = get_crop(30, 1344, 120)
    canvas.paste(_c30, (48, 738), _c30)
except Exception:
    pass
layout["financial_education"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/31_icon_Popular.png
try:
    _c31 = get_crop(31, 97, 108)
    canvas.paste(_c31, (34, 402), _c31)
except Exception:
    pass
layout["Popular"] = [34, 402, 131, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/32_text_Popular.png
try:
    _c32 = get_crop(32, 221, 78)
    canvas.paste(_c32, (44, 298), _c32)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/33_text_education_conference.png
try:
    _c33 = get_crop(33, 1344, 120)
    canvas.paste(_c33, (48, 378), _c33)
except Exception:
    pass
layout["education_conference"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/34_text_education_technology.png
try:
    _c34 = get_crop(34, 1344, 120)
    canvas.paste(_c34, (48, 498), _c34)
except Exception:
    pass
layout["education_technology"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/35_text_education_workshops.png
try:
    _c35 = get_crop(35, 1344, 120)
    canvas.paste(_c35, (48, 618), _c35)
except Exception:
    pass
layout["education_workshops"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/36_text_education_fair.png
try:
    _c36 = get_crop(36, 267, 45)
    canvas.paste(_c36, (161, 910), _c36)
except Exception:
    pass
layout["education_fair"] = [161, 910, 428, 955]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/37_text_Events.png
try:
    _c37 = get_crop(37, 188, 61)
    canvas.paste(_c37, (45, 1026), _c37)
except Exception:
    pass
layout["Events"] = [45, 1026, 233, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/38_text_Sat_Jun_29.png
try:
    _c38 = get_crop(38, 191, 45)
    canvas.paste(_c38, (390, 2391), _c38)
except Exception:
    pass
layout["Sat,_Jun_29"] = [390, 2391, 581, 2436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/39_text_1I_00_AM_PDT.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 2305), _c39)
except Exception:
    pass
layout["1I:00_AM_PDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/40_text_Embracing_Inspiring_the_Future.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 2305), _c40)
except Exception:
    pass
layout["Embracing_&_Inspiring_the"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/41_text_Sun_Apr_28.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (288, 2804), _c41)
except Exception:
    pass
layout["Sun,_Apr_28"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/42_text_II_O0_AM_PDT.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (576, 2804), _c42)
except Exception:
    pass
layout["II:O0_AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/43_clickable_education_fair.png
try:
    _c43 = get_crop(43, 1344, 144)
    canvas.paste(_c43, (48, 858), _c43)
except Exception:
    pass
layout["education_fair"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_09_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-11/44_clickable_Home.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (0, 2804), _c44)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
