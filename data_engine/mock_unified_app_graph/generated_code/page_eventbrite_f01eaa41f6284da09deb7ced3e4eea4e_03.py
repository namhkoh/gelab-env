# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_03
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5.png
# step_index: 3/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 72)], fill="#bdbdbd")
draw.line([(0, 72), (1440, 72)], fill="#a9a9a9", width=1)

# Header area (kept white, but draw accent underline)
underline_x0 = 48
underline_x1 = 1392
underline_y = 172
draw.line([(underline_x0, underline_y), (underline_x1, underline_y)], fill="#2f54d9", width=6)

# Light divider under header
draw.line([(0, underline_y + 6), (1440, underline_y + 6)], fill="#efefef", width=1)

# "Popular" list separators (between the popular items)
popular_item_tops = [378, 498, 618, 738, 858]
separator_offsets = 84  # approximate item height for separators
for y_top in popular_item_tops:
    sep_y = y_top + separator_offsets
    draw.line([(48, sep_y), (1392, sep_y)], fill="#eceff3", width=1)

# Divider above Events section
events_divider_y = 1026
draw.line([(48, events_divider_y), (1392, events_divider_y)], fill="#e6e6e9", width=1)

# Event cards background (rounded rectangles) using detected card positions
event_cards = [
    (48, 1117, 1344, 396),
    (48, 1513, 1344, 396),
    (48, 1909, 1344, 396),
    (48, 2305, 1344, 396),
]
for (x, y, w, h) in event_cards:
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    # slightly inset the fill to create a faint outline gap from the page edge
    inset = 0
    r = 12
    draw.rounded_rectangle([(x1 + inset, y1 + inset), (x2 - inset, y2 - inset)],
                           radius=r, fill="#ffffff", outline="#e9e9eb", width=1)

    # subtle bottom separator for each card
    draw.line([(x1 + 12, y2 - 1), (x2 - 12, y2 - 1)], fill="#f2f2f4", width=1)

# Subtle separators between stacked cards (in case layout has margin)
for _, y, _, h in event_cards:
    boundary_y = y - 20
    if 80 < boundary_y < 2900:
        draw.line([(48, boundary_y), (1392, boundary_y)], fill="#fbfbfc", width=1)

# Bottom navigation bar background and top border
bottom_nav_y0 = 2804
bottom_nav_y1 = 2960
draw.rectangle([(0, bottom_nav_y0), (1440, bottom_nav_y1)], fill="#ffffff")
draw.line([(0, bottom_nav_y0), (1440, bottom_nav_y0)], fill="#e6e6e9", width=1)

# Left page margin guideline (very light) and right margin guideline
draw.line([(48, 72), (48, 2760)], fill="#fbfbfc", width=1)
draw.line([(1392, 72), (1392, 2760)], fill="#fbfbfc", width=1)

# Overall subtle page tint (very light) to match screenshot's soft white background
# (draw a very large translucent rectangle by layering slightly off-white)
try:
    # If draw supports RGBA via tuple with alpha in pillow environment
    draw.rectangle([(0, 72), (1440, 2960)], fill=(250, 251, 253, 12))
except Exception:
    # fallback: draw a barely different white field
    pass

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/00_icon_Sports.png
try:
    _c0 = get_crop(0, 1344, 191)
    canvas.paste(_c0, (48, 72), _c0)
except Exception:
    pass
layout["Sports"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/01_icon_4.35.png
try:
    _c1 = get_crop(1, 122, 105)
    canvas.paste(_c1, (56, 118), _c1)
except Exception:
    pass
layout["4.35"] = [56, 118, 178, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 56, 59)
    canvas.paste(_c2, (313, 4), _c2)
except Exception:
    pass
layout["icon_2"] = [313, 4, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/03_icon_Basement_Berkeleyl.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1909), _c3)
except Exception:
    pass
layout["Basement_Berkeleyl"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/04_icon_4.35.png
try:
    _c4 = get_crop(4, 56, 61)
    canvas.paste(_c4, (115, 3), _c4)
except Exception:
    pass
layout["4.35"] = [115, 3, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/05_icon_4.35.png
try:
    _c5 = get_crop(5, 53, 59)
    canvas.paste(_c5, (183, 3), _c5)
except Exception:
    pass
layout["4.35"] = [183, 3, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 42, 54)
    canvas.paste(_c6, (254, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [254, 6, 296, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/07_icon_Voodoo_Combat_Gym.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1117), _c7)
except Exception:
    pass
layout["Voodoo_Combat_Gym"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/08_icon_Events.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 1117), _c8)
except Exception:
    pass
layout["Events"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/09_icon_Sat_Jun_15_._9_00_AM_PDT.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (288, 2804), _c9)
except Exception:
    pass
layout["Sat,_Jun_15_._9:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (864, 2804), _c10)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/11_icon_Sat_Jun_15_._9_00_AM_PDT.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (576, 2804), _c11)
except Exception:
    pass
layout["Sat,_Jun_15_._9:00_AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/12_icon_Sports_Basement_Berkeley.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1513), _c12)
except Exception:
    pass
layout["Sports_Basement_Berkeley"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/13_icon_Cancel.png
try:
    _c13 = get_crop(13, 47, 60)
    canvas.paste(_c13, (1322, 2), _c13)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 92, 63)
    canvas.paste(_c14, (1216, 0), _c14)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1308, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/15_icon_Sat_May_11.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 2305), _c15)
except Exception:
    pass
layout["Sat,_May_11"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/16_icon_Oakland_Ballers_Meet_Greet_at_Sports.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1909), _c16)
except Exception:
    pass
layout["Oakland_Ballers_Meet_&_Gr"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/17_icon_More.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (1152, 2804), _c17)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1099, 96), _c18)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/19_icon_Home.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (0, 2804), _c19)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/20_icon_B.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1909), _c20)
except Exception:
    pass
layout["B'"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/21_icon_4.35.png
try:
    _c21 = get_crop(21, 92, 60)
    canvas.paste(_c21, (15, 3), _c21)
except Exception:
    pass
layout["4.35"] = [15, 3, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 89, 93)
    canvas.paste(_c22, (35, 768), _c22)
except Exception:
    pass
layout["icon_22"] = [35, 768, 124, 861]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 149, 144)
    canvas.paste(_c23, (1243, 97), _c23)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 93, 93)
    canvas.paste(_c24, (33, 647), _c24)
except Exception:
    pass
layout["icon_24"] = [33, 647, 126, 740]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/25_icon_Events.png
try:
    _c25 = get_crop(25, 85, 84)
    canvas.paste(_c25, (37, 892), _c25)
except Exception:
    pass
layout["Events"] = [37, 892, 122, 976]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/26_icon_sports_events.png
try:
    _c26 = get_crop(26, 94, 94)
    canvas.paste(_c26, (32, 529), _c26)
except Exception:
    pass
layout["sports_events"] = [32, 529, 126, 623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/27_icon_B.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1513), _c27)
except Exception:
    pass
layout["B'"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/28_text_Popular.png
try:
    _c28 = get_crop(28, 221, 78)
    canvas.paste(_c28, (44, 298), _c28)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/29_text_sports_basement.png
try:
    _c29 = get_crop(29, 1344, 120)
    canvas.paste(_c29, (48, 378), _c29)
except Exception:
    pass
layout["sports_basement"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/30_text_sports_events.png
try:
    _c30 = get_crop(30, 1344, 120)
    canvas.paste(_c30, (48, 498), _c30)
except Exception:
    pass
layout["sports_events"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/31_text_sports_and_fitness.png
try:
    _c31 = get_crop(31, 1344, 120)
    canvas.paste(_c31, (48, 618), _c31)
except Exception:
    pass
layout["sports_and_fitness"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/32_text_sports_networking.png
try:
    _c32 = get_crop(32, 1344, 120)
    canvas.paste(_c32, (48, 738), _c32)
except Exception:
    pass
layout["sports_networking"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/33_text_sportsdrome_speedway.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 858), _c33)
except Exception:
    pass
layout["sportsdrome_speedway"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/34_text_Events.png
try:
    _c34 = get_crop(34, 186, 57)
    canvas.paste(_c34, (46, 1029), _c34)
except Exception:
    pass
layout["Events"] = [46, 1029, 232, 1086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/35_text_Sat_May_11.png
try:
    _c35 = get_crop(35, 207, 45)
    canvas.paste(_c35, (390, 2361), _c35)
except Exception:
    pass
layout["Sat,_May_11"] = [390, 2361, 597, 2406]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/36_text_9_00_AM_PDT.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 2305), _c36)
except Exception:
    pass
layout["9:00_AM_PDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/37_text_Birding_walk_with_JT_Birds_and_Nocs.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2305), _c37)
except Exception:
    pass
layout["Birding_walk_with_JT_Bird"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/38_text_Sports_Basement_Presidio.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 2305), _c38)
except Exception:
    pass
layout["Sports_Basement_Presidio"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/39_text_8_59_creator_followers.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 2305), _c39)
except Exception:
    pass
layout["8_59_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_03_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-5/40_text_Sat_Jun_15_._9_00_AM_PDT.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (576, 2804), _c40)
except Exception:
    pass
layout["Sat,_Jun_15_._9:00_AM_PDT"] = [576, 2804, 864, 2960]
