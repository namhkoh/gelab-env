# page_id: page_eventbrite_fc2bf7bdcfa6451293535f72ef3fd795_08
# screenshot: 2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10.png
# step_index: 8/8
# task: Open Eventbrite. Search for events by 'Music' under online events. Choose the second event in the list. Get the event's duration information.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area at top (~108px tall)
status_h = 108
draw.rectangle((0, 0, 1440, status_h), fill="#E6E6E6")

# Hero image background area (dark band behind hero image)
hero_top = status_h
hero_bottom = 700
draw.rectangle((0, hero_top, 1440, hero_bottom), fill="#111214")

# Subtle bottom fade for hero (simulate a soft divider into content)
fade_height = 18
for i in range(fade_height):
    alpha = int(10 + (i / fade_height) * 20)  # just control darkness steps
    shade = (10 + i * 4)
    hexcol = "#{:02x}{:02x}{:02x}".format(shade, shade, shade)
    draw.rectangle((0, hero_bottom - fade_height + i, 1440, hero_bottom - fade_height + i + 1), fill=hexcol)

# Main content area background (keeps white, but add a very slight warm tint band under hero)
content_band_top = hero_bottom
content_band_bottom = hero_bottom + 60
draw.rectangle((0, content_band_top, 1440, content_band_bottom), fill="#FBFAFB")

# Organizer card (rounded rectangle behind organizer info + Follow button)
card_x0, card_y0 = 40, 920
card_x1, card_y1 = 1400, 1040
draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1), radius=28, fill="#F6F5FA", outline="#ECE9F0", width=1)

# Small subtle drop shadow under the card
shadow_top = card_y1
for s in range(6):
    opacity = int(6 - s)
    shade = 230 + s  # slightly darker
    draw.rectangle((card_x0 + 2, shadow_top + s, card_x1 - 2, shadow_top + s + 1), fill="#F0EFF1")

# Section separators (thin rules between major sections)
sep_color = "#E9E7EA"
draw.line((48, 1160, 1392, 1160), fill=sep_color, width=1)   # after organizer + meta
draw.line((46, 1600, 1394, 1600), fill=sep_color, width=1)   # after refund policy area
draw.line((46, 2040, 1394, 2040), fill=sep_color, width=1)   # header for about section

# "About this event" section rounded background (slightly tinted block behind the tags area)
about_block_x0, about_block_y0 = 40, 1880
about_block_x1, about_block_y1 = 1400, 2300
draw.rectangle((about_block_x0, about_block_y0, about_block_x1, about_block_y1), fill="#FFFFFF")

# Soft rounded pill background where category tag sits (do not draw any text)
pill_x0, pill_y0 = 40, 1960
pill_x1, pill_y1 = 260, 2010
draw.rounded_rectangle((pill_x0, pill_y0, pill_x1, pill_y1), radius=22, fill="#F0EFF3", outline=None)

# Light divider above sticky bottom ticket bar
bottom_bar_h = 200
bottom_bar_top = 2960 - bottom_bar_h
draw.line((0, bottom_bar_top, 1440, bottom_bar_top), fill="#E6E3E8", width=1)

# Sticky bottom ticket bar background
draw.rectangle((0, bottom_bar_top, 1440, 2960), fill="#FBF7F6")

# Subtle inset panel for the left price area (do not draw the actual price text)
price_panel_x0, price_panel_y0 = 40, bottom_bar_top + 24
price_panel_x1, price_panel_y1 = 520, 2960 - 24
draw.rounded_rectangle((price_panel_x0, price_panel_y0, price_panel_x1, price_panel_y1), radius=12, fill="#FFFFFF", outline="#E8E6EA", width=1)

# Subtle shadow line above the ticket button area (right side)
ticket_panel_x0, ticket_panel_y0 = 540, bottom_bar_top + 16
ticket_panel_x1, ticket_panel_y1 = 1400, 2960 - 16
# draw a light rectangle background to indicate button area region (actual button will be pasted)
draw.rectangle((ticket_panel_x0, ticket_panel_y0, ticket_panel_x1, ticket_panel_y1), fill="#FFFFFF")

# Top toolbar divider under status/hero for a clean separation
draw.line((0, hero_bottom, 1440, hero_bottom), fill="#0c0c0c", width=1)

# Final subtle full-width shadow near bottom to anchor bar
for i in range(3):
    y = bottom_bar_top + i
    shade = 235 - i * 3
    draw.line((0, y, 1440, y), fill=(shade, shade, shade), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1321), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1321, 1344, 1465]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/02_icon_INFINITUM_QUANTUMISTEPHANY.png
try:
    _c2 = get_crop(2, 773, 144)
    canvas.paste(_c2, (144, 1280), _c2)
except Exception:
    pass
layout["INFINITUM_QUANTUMISTEPHAN"] = [144, 1280, 917, 1424]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/03_icon_More.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/05_icon_8.05.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["8.05"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/06_icon_Going_fast.png
try:
    _c6 = get_crop(6, 334, 86)
    canvas.paste(_c6, (41, 753), _c6)
except Exception:
    pass
layout["Going_fast"] = [41, 753, 375, 839]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/07_icon_8.05.png
try:
    _c7 = get_crop(7, 67, 72)
    canvas.paste(_c7, (178, 0), _c7)
except Exception:
    pass
layout["8.05"] = [178, 0, 245, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/08_icon_8.05.png
try:
    _c8 = get_crop(8, 64, 71)
    canvas.paste(_c8, (113, 0), _c8)
except Exception:
    pass
layout["8.05"] = [113, 0, 177, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 54, 64)
    canvas.paste(_c9, (1318, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1318, 1, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 69, 71)
    canvas.paste(_c10, (306, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [306, 0, 375, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 56, 71)
    canvas.paste(_c11, (246, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [246, 0, 302, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 42, 63)
    canvas.paste(_c12, (1273, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1273, 2, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 59)
    canvas.paste(_c13, (1216, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [1216, 5, 1272, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/14_icon_Music.png
try:
    _c14 = get_crop(14, 308, 99)
    canvas.paste(_c14, (41, 2227), _c14)
except Exception:
    pass
layout["Music"] = [41, 2227, 349, 2326]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 71)
    canvas.paste(_c15, (382, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 0, 434, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/16_icon_The_organizer_will_review_refund_request.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1578), _c16)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1578, 1392, 1722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/17_text_RNB_WIBES_CLT.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1116, 108), _c17)
except Exception:
    pass
layout["RNB_WIBES_CLT"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/18_text_Saturday.png
try:
    _c18 = get_crop(18, 251, 73)
    canvas.paste(_c18, (38, 887), _c18)
except Exception:
    pass
layout["Saturday,"] = [38, 887, 289, 960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/19_text_11.png
try:
    _c19 = get_crop(19, 64, 52)
    canvas.paste(_c19, (407, 895), _c19)
except Exception:
    pass
layout["11"] = [407, 895, 471, 947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/20_text_3.30_PM.png
try:
    _c20 = get_crop(20, 209, 56)
    canvas.paste(_c20, (511, 893), _c20)
except Exception:
    pass
layout["3.30_PM"] = [511, 893, 720, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/21_text_RNB_VIBES_reloaded_OASIS_FIRST_R_B.png
try:
    _c21 = get_crop(21, 773, 144)
    canvas.paste(_c21, (144, 1280), _c21)
except Exception:
    pass
layout["RNB_VIBES_reloaded@OASIS_"] = [144, 1280, 917, 1424]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/22_text_INDOOR_FESTIVAL_EVERIII.png
try:
    _c22 = get_crop(22, 773, 144)
    canvas.paste(_c22, (144, 1280), _c22)
except Exception:
    pass
layout["INDOOR_FESTIVAL_EVERIII"] = [144, 1280, 917, 1424]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/23_text_Online_event.png
try:
    _c23 = get_crop(23, 274, 52)
    canvas.paste(_c23, (139, 1626), _c23)
except Exception:
    pass
layout["Online_event"] = [139, 1626, 413, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/24_text_About_this_event.png
try:
    _c24 = get_crop(24, 452, 56)
    canvas.paste(_c24, (46, 2142), _c24)
except Exception:
    pass
layout["About_this_event"] = [46, 2142, 498, 2198]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/25_text_R_B_VIBES_Reloaded.png
try:
    _c25 = get_crop(25, 387, 49)
    canvas.paste(_c25, (44, 2373), _c25)
except Exception:
    pass
layout["R&B_VIBES_Reloaded"] = [44, 2373, 431, 2422]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/26_text_Ist_ever_indoor.png
try:
    _c26 = get_crop(26, 282, 43)
    canvas.paste(_c26, (47, 2489), _c26)
except Exception:
    pass
layout["Ist_ever_indoor"] = [47, 2489, 329, 2532]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/27_text_Music_FESTIVAL.png
try:
    _c27 = get_crop(27, 302, 51)
    canvas.paste(_c27, (46, 2539), _c27)
except Exception:
    pass
layout["Music_FESTIVAL"] = [46, 2539, 348, 2590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/28_text_Same_excellent_vibes_from_SouthPark_mall.png
try:
    _c28 = get_crop(28, 570, 144)
    canvas.paste(_c28, (822, 2768), _c28)
except Exception:
    pass
layout["Same_excellent_vibes_from"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_08_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-10/29_text_S0_-_S450.png
try:
    _c29 = get_crop(29, 228, 61)
    canvas.paste(_c29, (89, 2811), _c29)
except Exception:
    pass
layout["S0_-_S450"] = [89, 2811, 317, 2872]
