# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_09
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11.png
# step_index: 9/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw UI background and structure for Event page (canvas and draw provided)

# Status bar (top)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#aeb3aa")

# Notification banner (light mint) below status bar
banner_top = status_h + 8
banner_bottom = banner_top + 92
draw.rectangle([(0, banner_top), (1440, banner_bottom)], fill="#e9f6ee")

# Thin divider under banner
draw.line([(24, banner_bottom + 6), (1440 - 24, banner_bottom + 6)], fill="#d9d9d9", width=1)

# Hero image / top artwork area (dark banner)
hero_top = banner_bottom + 6
hero_bottom = hero_top + 260
# simple vertical gradient-ish fill (dark to darker)
for i in range(hero_top, hero_bottom):
    # interpolate between two dark colors
    t = (i - hero_top) / max(1, (hero_bottom - hero_top))
    r = int((11 * (1 - t)) + (20 * t))
    g = int((12 * (1 - t)) + (24 * t))
    b = int((15 * (1 - t)) + (40 * t))
    draw.line([(0, i), (1440, i)], fill=(r, g, b))

# Soft top edge shadow below hero
draw.rectangle([(0, hero_bottom - 6), (1440, hero_bottom)], fill=(0, 0, 0, 64))

# Main content background (keeps white, but add subtle warm tint to large content area)
content_top = hero_bottom + 18
# subtle large content background is same white; add an overall very subtle off-white band where cards sit
draw.rectangle([(0, content_top), (1440, 2600)], fill="#ffffff")

# Organizer info card (rounded rectangle) behind organizer profile and follow button
card_left = 32
card_right = 1408
card_top = 1108
card_bottom = card_top + 150
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                       radius=28, fill="#fbf9fc", outline="#eeeaf2", width=1)

# Add a subtle inner shadow / highlight for the organizer card (top highlight)
draw.line([(card_left + 8, card_top + 8), (card_right - 8, card_top + 8)], fill="#ffffff", width=1)

# Section separators and subtle dividers
# Divider under details (e.g., under refund policy / details block)
divider_y1 = 1700
draw.line([(40, divider_y1), (1440 - 40, divider_y1)], fill="#ededf0", width=1)

# Divider before "About this event" section (lighter)
divider_y2 = 1920
draw.line([(24, divider_y2), (1440 - 24, divider_y2)], fill="#f0eff2", width=1)

# About section background pill area backdrop (subtle)
about_pill_bg_top = 2020
about_pill_bg_bottom = about_pill_bg_top + 64
# this area remains white; draw a faint rounded rect behind tags area to anchor it visually
draw.rounded_rectangle([(40, about_pill_bg_top), (600, about_pill_bg_top + 48)],
                       radius=24, fill="#fafbfd", outline=None)

# Divider above Location section
loc_div_y = 2460
draw.line([(24, loc_div_y), (1440 - 24, loc_div_y)], fill="#efeef1", width=1)

# Location background card separator area (light)
loc_card_top = loc_div_y + 20
draw.rectangle([(0, loc_card_top), (1440, loc_card_top + 220)], fill="#ffffff")

# Bottom sticky ticket bar
bottom_bar_top = 2720
bottom_bar_bottom = 2960
draw.rectangle([(0, bottom_bar_top), (1440, bottom_bar_bottom)], fill="#faf9fb")

# Top hairline shadow for bottom bar
draw.line([(0, bottom_bar_top), (1440, bottom_bar_top)], fill="#e8e6ea", width=1)

# Left price area background within bottom bar (subtle)
price_pad = 28
draw.rectangle([(price_pad, bottom_bar_top + 20), (420, bottom_bar_bottom - 20)], fill="#faf9fb", outline=None)

# Right side area (where Get tickets button will be pasted) - draw subtle bounding bank color band behind it
get_button_band_left = 520
get_button_band_right = 1440 - 32
band_top = bottom_bar_top + 20
band_bottom = bottom_bar_bottom - 20
draw.rounded_rectangle([(get_button_band_left, band_top), (get_button_band_right, band_bottom)],
                       radius=12, fill="#f6f3f1", outline=None)

# Small horizontal separators for content blocks further up
draw.line([(24, 1560), (1440 - 24, 1560)], fill="#f3f2f4", width=1)
draw.line([(24, 1840), (1440 - 24, 1840)], fill="#f4f3f5", width=1)

# Subtle left vertical margin guide (visual only)
draw.line([(40, content_top), (40, bottom_bar_top)], fill="#ffffff", width=2)

# End of background/structure rendering

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/02_icon_Celebrate_Art_at_San_Francisco_s_Premier.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2332), _c2)
except Exception:
    pass
layout["Celebrate_Art_at_San_Fran"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/03_icon_4.33.png
try:
    _c3 = get_crop(3, 61, 63)
    canvas.paste(_c3, (180, 1), _c3)
except Exception:
    pass
layout["4.33"] = [180, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 62, 61)
    canvas.paste(_c4, (309, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [309, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/05_icon_4.33.png
try:
    _c5 = get_crop(5, 63, 65)
    canvas.paste(_c5, (112, 0), _c5)
except Exception:
    pass
layout["4.33"] = [112, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/06_icon_Dismiss_notification.png
try:
    _c6 = get_crop(6, 142, 142)
    canvas.paste(_c6, (1251, 97), _c6)
except Exception:
    pass
layout["Dismiss_notification"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/07_icon_Performing_Visual_Arts.png
try:
    _c7 = get_crop(7, 234, 144)
    canvas.paste(_c7, (48, 2332), _c7)
except Exception:
    pass
layout["Performing_&_Visual_Arts"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 56)
    canvas.paste(_c8, (250, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [250, 5, 299, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 66)
    canvas.paste(_c9, (1317, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1317, 0, 1373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 79, 65)
    canvas.paste(_c10, (1212, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1212, 0, 1291, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/11_icon_Ticket_sales_end_soon.png
try:
    _c11 = get_crop(11, 547, 84)
    canvas.paste(_c11, (40, 753), _c11)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/12_icon_Shipyard_Trust_for_the_Arts.png
try:
    _c12 = get_crop(12, 558, 144)
    canvas.paste(_c12, (288, 1155), _c12)
except Exception:
    pass
layout["Shipyard_Trust_for_the_Ar"] = [288, 1155, 846, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 64)
    canvas.paste(_c13, (382, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [382, 1, 434, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 43, 62)
    canvas.paste(_c14, (1272, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1272, 2, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/15_icon_Show_map.png
try:
    _c15 = get_crop(15, 226, 144)
    canvas.paste(_c15, (1166, 2550), _c15)
except Exception:
    pass
layout["Show_map"] = [1166, 2550, 1392, 2694]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/16_icon_4.33.png
try:
    _c16 = get_crop(16, 92, 62)
    canvas.paste(_c16, (15, 2), _c16)
except Exception:
    pass
layout["4.33"] = [15, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/17_icon_SHIPYARD.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1116, 108), _c17)
except Exception:
    pass
layout["SHIPYARD"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/18_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (36, 108), _c18)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/19_text_Saturday_April_27.png
try:
    _c19 = get_crop(19, 451, 77)
    canvas.paste(_c19, (38, 885), _c19)
except Exception:
    pass
layout["Saturday;_April_27"] = [38, 885, 489, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/20_text_1I_00AM.png
try:
    _c20 = get_crop(20, 241, 56)
    canvas.paste(_c20, (523, 893), _c20)
except Exception:
    pass
layout["1I:00AM"] = [523, 893, 764, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/21_text_Shipyard_Open_Studios.png
try:
    _c21 = get_crop(21, 558, 144)
    canvas.paste(_c21, (288, 1155), _c21)
except Exception:
    pass
layout["Shipyard_Open_Studios"] = [288, 1155, 846, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/22_text_Spring_2024.png
try:
    _c22 = get_crop(22, 331, 144)
    canvas.paste(_c22, (1013, 1195), _c22)
except Exception:
    pass
layout["Spring_2024"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/23_text_Hunters_Point_Shipyard.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1422), _c23)
except Exception:
    pass
layout["Hunters_Point_Shipyard"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/24_text_days_7_hrs.png
try:
    _c24 = get_crop(24, 228, 63)
    canvas.paste(_c24, (172, 1577), _c24)
except Exception:
    pass
layout["days_7_hrs"] = [172, 1577, 400, 1640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/25_text_Refund_policy.png
try:
    _c25 = get_crop(25, 299, 63)
    canvas.paste(_c25, (138, 1685), _c25)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/26_text_The_organizer_will_review_refund_request.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1422), _c26)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/27_text_About_this_event.png
try:
    _c27 = get_crop(27, 453, 65)
    canvas.paste(_c27, (44, 1982), _c27)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 497, 2047]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/28_text_Location.png
try:
    _c28 = get_crop(28, 246, 63)
    canvas.paste(_c28, (41, 2594), _c28)
except Exception:
    pass
layout["Location"] = [41, 2594, 287, 2657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/29_text_S0_-_25.png
try:
    _c29 = get_crop(29, 198, 61)
    canvas.paste(_c29, (89, 2811), _c29)
except Exception:
    pass
layout["S0_-_$25"] = [89, 2811, 287, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_09_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-11/30_clickable_Organizer_profile_picture.png
try:
    _c30 = get_crop(30, 144, 144)
    canvas.paste(_c30, (96, 1194), _c30)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1194, 240, 1338]
