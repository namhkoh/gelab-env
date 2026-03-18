# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_08
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10.png
# step_index: 8/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 56
draw.rectangle([(0, 0), (canvas.width, status_h)], fill="#cfcfcf")

# Header / toolbar area (search bar area)
header_top = status_h
header_h = 94
draw.rectangle([(0, header_top), (canvas.width, header_top + header_h)], fill="#ffffff")

# Blue underline below the search header
underline_y = header_top + header_h - 2
draw.rectangle([(48, underline_y), (canvas.width - 48, underline_y + 4)], fill="#2E5BFF")

# subtle divider under header
draw.line([(0, underline_y + 6), (canvas.width, underline_y + 6)], fill="#e6e6ea", width=1)

# "Popular" section background (keep white but add a faint vertical padding background strip)
popular_top = 280
popular_bottom = 960
draw.rectangle([(0, popular_top - 20), (canvas.width, popular_bottom)], fill="#ffffff")

# separators between popular list items (positions inferred from layout)
popular_seps = [378 + i * 120 for i in range(1, 5)]  # ~498,618,738,858
for y in popular_seps:
    draw.line([(48, y), (canvas.width - 48, y)], fill="#f0f0f3", width=1)

# "Events" heading area divider
events_heading_y = 1025
draw.line([(48, events_heading_y + 72), (canvas.width - 48, events_heading_y + 72)], fill="#f0f0f3", width=1)

# Event list card backgrounds (rounded rectangles behind each event group)
event_cards = [
    (48, 1117, 48 + 1344, 1117 + 396),
    (48, 1513, 48 + 1344, 1513 + 396),
    (48, 1909, 48 + 1344, 1909 + 396),
    (48, 2305, 48 + 1344, 2305 + 396)
]
for (x1, y1, x2, y2) in event_cards:
    # card background
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=12, fill="#fbfbfd", outline="#edeef2", width=1)
    # subtle inner top divider to separate thumbnail/title area visually
    draw.line([(x1 + 12, y1 + 94), (x2 - 12, y1 + 94)], fill="#f3f3f6", width=1)

# Thin separators between event cards
for (x1, y1, x2, y2) in event_cards:
    draw.line([(x1 + 12, y2 + 8), (x2 - 12, y2 + 8)], fill="#f1f1f4", width=1)

# Bottom navigation bar area separator and background
nav_top = 2804
draw.line([(0, nav_top), (canvas.width, nav_top)], fill="#e7e7ea", width=1)
draw.rectangle([(0, nav_top), (canvas.width, canvas.height)], fill="#ffffff")

# Slight left and right edge gutters (visual alignment guides - very subtle)
draw.rectangle([(0, 0), (12, canvas.height)], fill="#ffffff", outline=None)
draw.rectangle([(canvas.width - 12, 0), (canvas.width, canvas.height)], fill="#ffffff", outline=None)

# Final subtle overall vignette/light shadow at major section boundaries
boundaries = [popular_top - 10, events_heading_y, nav_top]
for by in boundaries:
    draw.line([(48, by), (canvas.width - 48, by)], fill="#fafafa", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/00_icon_ESS.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1513), _c0)
except Exception:
    pass
layout["ESS"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/01_icon_EEM_Ltd.png
try:
    _c1 = get_crop(1, 133, 50)
    canvas.paste(_c1, (391, 1354), _c1)
except Exception:
    pass
layout["EEM_Ltd"] = [391, 1354, 524, 1404]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/02_icon_BASH.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1909), _c2)
except Exception:
    pass
layout["BASH"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/03_icon_Business.png
try:
    _c3 = get_crop(3, 58, 60)
    canvas.paste(_c3, (312, 3), _c3)
except Exception:
    pass
layout["Business"] = [312, 3, 370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/04_icon_Business.png
try:
    _c4 = get_crop(4, 1344, 191)
    canvas.paste(_c4, (48, 72), _c4)
except Exception:
    pass
layout["Business"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/05_icon_8.02.png
try:
    _c5 = get_crop(5, 55, 60)
    canvas.paste(_c5, (182, 3), _c5)
except Exception:
    pass
layout["8.02"] = [182, 3, 237, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/06_icon_8.02.png
try:
    _c6 = get_crop(6, 57, 62)
    canvas.paste(_c6, (115, 2), _c6)
except Exception:
    pass
layout["8.02"] = [115, 2, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 57)
    canvas.paste(_c7, (252, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [252, 5, 297, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/08_icon_8.02.png
try:
    _c8 = get_crop(8, 115, 105)
    canvas.paste(_c8, (59, 118), _c8)
except Exception:
    pass
layout["8.02"] = [59, 118, 174, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 68)
    canvas.paste(_c9, (1157, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1157, 0, 1200, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/10_icon_Cancel.png
try:
    _c10 = get_crop(10, 95, 66)
    canvas.paste(_c10, (1216, 0), _c10)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1311, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/11_icon_Businesses_in_Bolsover_District.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 1117), _c11)
except Exception:
    pass
layout["Businesses_in_Bolsover_Di"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/12_icon_Sat_May_18_._1I_00_AM_PDT.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1909), _c12)
except Exception:
    pass
layout["Sat,_May_18_._1I:00_AM_PD"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/13_icon_6_00_PM_PDT.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 2305), _c13)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/14_icon_Black_Business_CoOp_Monthly_Business.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1513), _c14)
except Exception:
    pass
layout["Black_Business_CoOp_Month"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 51, 66)
    canvas.paste(_c15, (1320, 0), _c15)
except Exception:
    pass
layout["Cancel"] = [1320, 0, 1371, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/16_icon_Crl.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (864, 2804), _c16)
except Exception:
    pass
layout["Crl"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/17_icon_Sat_May_18_._1I_00_AM_PDT.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1909), _c17)
except Exception:
    pass
layout["Sat,_May_18_._1I:00_AM_PD"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1099, 96), _c18)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/19_icon_SmallBizSootlizht_net.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["SmallBizSootlizht_net"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/20_icon_8_00_AM_GMT_O1_O0.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1117), _c20)
except Exception:
    pass
layout["8:00_AM_GMT+O1:O0"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/21_icon_Cancel.png
try:
    _c21 = get_crop(21, 149, 144)
    canvas.paste(_c21, (1243, 97), _c21)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/22_icon_small_business.png
try:
    _c22 = get_crop(22, 96, 97)
    canvas.paste(_c22, (31, 528), _c22)
except Exception:
    pass
layout["small_business"] = [31, 528, 127, 625]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/23_icon_Business.png
try:
    _c23 = get_crop(23, 48, 63)
    canvas.paste(_c23, (383, 2), _c23)
except Exception:
    pass
layout["Business"] = [383, 2, 431, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/24_icon_SMALL_BIZ.png
try:
    _c24 = get_crop(24, 318, 133)
    canvas.paste(_c24, (42, 2427), _c24)
except Exception:
    pass
layout["SMALL_BIZ"] = [42, 2427, 360, 2560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 88, 92)
    canvas.paste(_c25, (36, 648), _c25)
except Exception:
    pass
layout["icon_25"] = [36, 648, 124, 740]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/26_icon_Hampton_Inn_Suites_LAX_El_Segundo.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Hampton_Inn_&_Suites_LAX_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/27_text_8.02.png
try:
    _c27 = get_crop(27, 91, 43)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["8.02"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/28_text_Popular.png
try:
    _c28 = get_crop(28, 221, 78)
    canvas.paste(_c28, (44, 298), _c28)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/29_text_business_networking.png
try:
    _c29 = get_crop(29, 1344, 120)
    canvas.paste(_c29, (48, 378), _c29)
except Exception:
    pass
layout["business_networking"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/30_text_small_business.png
try:
    _c30 = get_crop(30, 1344, 120)
    canvas.paste(_c30, (48, 498), _c30)
except Exception:
    pass
layout["small_business"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/31_text_women_in_business.png
try:
    _c31 = get_crop(31, 1344, 120)
    canvas.paste(_c31, (48, 618), _c31)
except Exception:
    pass
layout["women_in_business"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/32_text_business_conference.png
try:
    _c32 = get_crop(32, 1344, 120)
    canvas.paste(_c32, (48, 738), _c32)
except Exception:
    pass
layout["business_conference"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/33_text_business_expo.png
try:
    _c33 = get_crop(33, 274, 52)
    canvas.paste(_c33, (162, 909), _c33)
except Exception:
    pass
layout["business_expo"] = [162, 909, 436, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/34_text_Events.png
try:
    _c34 = get_crop(34, 191, 67)
    canvas.paste(_c34, (44, 1025), _c34)
except Exception:
    pass
layout["Events"] = [44, 1025, 235, 1092]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/35_text_9_00_AM_PDT.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1513), _c35)
except Exception:
    pass
layout["9:00_AM_PDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/36_text_3365_E_Artesia_Blvd.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1513), _c36)
except Exception:
    pass
layout["3365_E_Artesia_Blvd"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/37_text_8_28_creator_followers.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1513), _c37)
except Exception:
    pass
layout["8_28_creator_followers"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/38_text_SmallbizSpetlizht_net.png
try:
    _c38 = get_crop(38, 148, 24)
    canvas.paste(_c38, (196, 2352), _c38)
except Exception:
    pass
layout["SmallbizSpetlizht_net"] = [196, 2352, 344, 2376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/39_text_Thu_May_16_-.png
try:
    _c39 = get_crop(39, 218, 48)
    canvas.paste(_c39, (389, 2390), _c39)
except Exception:
    pass
layout["Thu,_May_16_-"] = [389, 2390, 607, 2438]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/40_text_6_00_PM_PDT.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 2305), _c40)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/41_text_SmallBizSootlizht_net.png
try:
    _c41 = get_crop(41, 148, 27)
    canvas.paste(_c41, (196, 2574), _c41)
except Exception:
    pass
layout["SmallBizSootlizht_net"] = [196, 2574, 344, 2601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/42_text_Hampton_Inn_Suites_LAX_El_Segundo.png
try:
    _c42 = get_crop(42, 1344, 396)
    canvas.paste(_c42, (48, 2305), _c42)
except Exception:
    pass
layout["Hampton_Inn_&_Suites_LAX_"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/43_text_Smal_BizSpotl_ght_net.png
try:
    _c43 = get_crop(43, 147, 27)
    canvas.paste(_c43, (51, 2629), _c43)
except Exception:
    pass
layout["Smal_BizSpotl_ght_net"] = [51, 2629, 198, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/44_text_Crl.png
try:
    _c44 = get_crop(44, 39, 14)
    canvas.paste(_c44, (995, 2794), _c44)
except Exception:
    pass
layout["Crl"] = [995, 2794, 1034, 2808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/45_clickable_business_expo.png
try:
    _c45 = get_crop(45, 1344, 144)
    canvas.paste(_c45, (48, 858), _c45)
except Exception:
    pass
layout["business_expo"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/46_clickable_Home.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (0, 2804), _c46)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_08_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-10/47_clickable_More.png
try:
    _c47 = get_crop(47, 288, 156)
    canvas.paste(_c47, (1152, 2804), _c47)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
