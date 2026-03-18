# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_03
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5.png
# step_index: 3/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page

# Base background (slightly off-white to match app background)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar (top ~56px) - grey bar behind system icons
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#9E9E9E")

# Header / Search area (below status bar)
header_top = status_h
header_bottom = 140
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# Blue underline for search field (matches the app accent)
underline_y = header_bottom - 4
draw.rectangle([(48, underline_y), (1392, underline_y + 6)], fill="#1E62FF")

# Thin divider below header
draw.line([(0, header_bottom + 2), (1440, header_bottom + 2)], fill="#E6E6E6", width=1)

# "Popular" list separators (between keyword rows). These are structural separators only.
popular_separators = [360, 480, 600, 720, 840]
for y in popular_separators:
    draw.line([(48, y), (1392, y)], fill="#F0F0F3", width=1)

# Light vertical margin guide (not visible in final app, subtle background band to suggest margin)
draw.rectangle([(0, header_bottom + 8), (48, 1100)], fill="#FFFFFF")  # keep margin area white

# Event row card backgrounds (rounded rectangles). These are just the cards behind the event crops.
event_rows = [
    (48, 1117, 48 + 1344, 1117 + 396),
    (48, 1513, 48 + 1344, 1513 + 396),
    (48, 1909, 48 + 1344, 1909 + 396),
    (48, 2305, 48 + 1344, 2305 + 396),
]
for (x1, y1, x2, y2) in event_rows:
    # subtle shadow (offset)
    shadow_offset = 6
    shadow_box = (x1, y1 + shadow_offset, x2, y2 + shadow_offset)
    draw.rounded_rectangle(shadow_box, radius=8, fill="#F4F5F7")
    # main card
    draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill="#FFFFFF", outline="#ECECF0")

    # divider line at card bottom
    draw.line([(x1 + 12, y2), (x2 - 12, y2)], fill="#E8E8EB", width=1)

# Subtle separators between event cards (in case of tight stacking)
for (_, y1, _, y2) in event_rows:
    draw.line([(48, y1 - 1), (1392, y1 - 1)], fill="#F3F4F6", width=1)

# Content area wide divider above bottom navigation
nav_top = 2760
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E6E6", width=1)

# Bottom navigation bar background
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")

# Light top shadow for bottom nav (to separate from content)
draw.rectangle([(0, nav_top), (1440, nav_top + 6)], fill="#FBFBFC")

# Safe zone filler on left/right edges (subtle)
edge_band_w = 24
draw.rectangle([(0, 0), (edge_band_w, 2960)], fill="#FFFFFF")
draw.rectangle([(1440 - edge_band_w, 0), (1440, 2960)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/00_icon_Men.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2305), _c0)
except Exception:
    pass
layout["Men"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/01_icon_8_257_creator_followers.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1909), _c1)
except Exception:
    pass
layout["8_257_creator_followers"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/02_icon_Education.png
try:
    _c2 = get_crop(2, 1344, 191)
    canvas.paste(_c2, (48, 72), _c2)
except Exception:
    pass
layout["Education]"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/03_icon_Events.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1117), _c3)
except Exception:
    pass
layout["Events"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/04_icon_9_00_AM_PDT.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1513), _c4)
except Exception:
    pass
layout["9:00_AM_PDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/05_icon_Education.png
try:
    _c5 = get_crop(5, 56, 59)
    canvas.paste(_c5, (313, 4), _c5)
except Exception:
    pass
layout["Education]"] = [313, 4, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/06_icon_4.56.png
try:
    _c6 = get_crop(6, 53, 60)
    canvas.paste(_c6, (184, 3), _c6)
except Exception:
    pass
layout["4.56"] = [184, 3, 237, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 43, 54)
    canvas.paste(_c7, (253, 7), _c7)
except Exception:
    pass
layout["icon_7"] = [253, 7, 296, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/08_icon_EU.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 1909), _c8)
except Exception:
    pass
layout["EU"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/09_icon_4.56.png
try:
    _c9 = get_crop(9, 57, 61)
    canvas.paste(_c9, (115, 3), _c9)
except Exception:
    pass
layout["4.56"] = [115, 3, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (864, 2804), _c10)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/11_icon_4.56.png
try:
    _c11 = get_crop(11, 111, 102)
    canvas.paste(_c11, (61, 119), _c11)
except Exception:
    pass
layout["4.56"] = [61, 119, 172, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/12_icon_Sun_Apr_28.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (288, 2804), _c12)
except Exception:
    pass
layout["Sun,_Apr_28"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/13_icon_II_O0_AM_PDT.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (576, 2804), _c13)
except Exception:
    pass
layout["II:O0_AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/14_icon_Educational_professionals.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1117), _c14)
except Exception:
    pass
layout["Educational_professionals"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 47, 61)
    canvas.paste(_c15, (1322, 2), _c15)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/16_icon_Sat.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1513), _c16)
except Exception:
    pass
layout["Sat,"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 90, 63)
    canvas.paste(_c17, (1216, 0), _c17)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1306, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/18_icon_City_Club_LA.png
try:
    _c18 = get_crop(18, 201, 52)
    canvas.paste(_c18, (390, 2571), _c18)
except Exception:
    pass
layout["City_Club_LA"] = [390, 2571, 591, 2623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1099, 96), _c19)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 149, 144)
    canvas.paste(_c20, (1243, 97), _c20)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/21_icon_Ist_Nature-Based_Education_Summit.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1513), _c21)
except Exception:
    pass
layout["Ist_Nature-Based_Educatio"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/22_icon_Education.png
try:
    _c22 = get_crop(22, 46, 60)
    canvas.paste(_c22, (384, 4), _c22)
except Exception:
    pass
layout["Education]"] = [384, 4, 430, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 93, 98)
    canvas.paste(_c23, (33, 528), _c23)
except Exception:
    pass
layout["icon_23"] = [33, 528, 126, 626]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/24_icon_4.56.png
try:
    _c24 = get_crop(24, 90, 59)
    canvas.paste(_c24, (16, 4), _c24)
except Exception:
    pass
layout["4.56"] = [16, 4, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 84, 91)
    canvas.paste(_c25, (40, 769), _c25)
except Exception:
    pass
layout["icon_25"] = [40, 769, 124, 860]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/26_icon_Education_You_networking_event.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1117), _c26)
except Exception:
    pass
layout["Education?You_networking_"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 86, 93)
    canvas.paste(_c28, (37, 648), _c28)
except Exception:
    pass
layout["icon_28"] = [37, 648, 123, 741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/29_text_Popular.png
try:
    _c29 = get_crop(29, 221, 78)
    canvas.paste(_c29, (44, 298), _c29)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/30_text_education_conference.png
try:
    _c30 = get_crop(30, 1344, 120)
    canvas.paste(_c30, (48, 378), _c30)
except Exception:
    pass
layout["education_conference"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/31_text_education_technology.png
try:
    _c31 = get_crop(31, 1344, 120)
    canvas.paste(_c31, (48, 498), _c31)
except Exception:
    pass
layout["education_technology"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/32_text_education_workshops.png
try:
    _c32 = get_crop(32, 1344, 120)
    canvas.paste(_c32, (48, 618), _c32)
except Exception:
    pass
layout["education_workshops"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/33_text_financial_education.png
try:
    _c33 = get_crop(33, 1344, 120)
    canvas.paste(_c33, (48, 738), _c33)
except Exception:
    pass
layout["financial_education"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/34_text_education_fair.png
try:
    _c34 = get_crop(34, 267, 45)
    canvas.paste(_c34, (161, 910), _c34)
except Exception:
    pass
layout["education_fair"] = [161, 910, 428, 955]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/35_text_Events.png
try:
    _c35 = get_crop(35, 188, 61)
    canvas.paste(_c35, (45, 1026), _c35)
except Exception:
    pass
layout["Events"] = [45, 1026, 233, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/36_text_Sat_Jun_29.png
try:
    _c36 = get_crop(36, 191, 45)
    canvas.paste(_c36, (390, 2391), _c36)
except Exception:
    pass
layout["Sat,_Jun_29"] = [390, 2391, 581, 2436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/37_text_1I_00_AM_PDT.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2305), _c37)
except Exception:
    pass
layout["1I:00_AM_PDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/38_text_Embracing_Inspiring_the_Future.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 2305), _c38)
except Exception:
    pass
layout["Embracing_&_Inspiring_the"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/39_text_Sun_Apr_28.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (288, 2804), _c39)
except Exception:
    pass
layout["Sun,_Apr_28"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/40_text_II_O0_AM_PDT.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (576, 2804), _c40)
except Exception:
    pass
layout["II:O0_AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/41_clickable_education_fair.png
try:
    _c41 = get_crop(41, 1344, 144)
    canvas.paste(_c41, (48, 858), _c41)
except Exception:
    pass
layout["education_fair"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_03_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-5/42_clickable_Home.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (0, 2804), _c42)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
