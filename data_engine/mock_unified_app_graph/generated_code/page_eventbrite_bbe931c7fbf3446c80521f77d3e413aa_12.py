# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_12
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14.png
# step_index: 12/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Eventbrite-like page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

width, height = canvas.size

# Overall page background (very light off-white)
bg_color = (250, 250, 252)
draw.rectangle([0, 0, width, height], fill=bg_color)

# Status bar area at top (~84px high)
status_h = 84
status_color = (241, 241, 244)  # subtle light gray overlay
draw.rectangle([0, 0, width, status_h], fill=status_color)

# Hero image area (large dark banner under status bar)
hero_top = status_h
hero_bottom = 520
# simple vertical gradient for hero image placeholder
top_color = (64, 64, 64)
bottom_color = (28, 28, 28)
for i in range(hero_bottom - hero_top):
    blend = i / max(1, (hero_bottom - hero_top - 1))
    r = int(top_color[0] * (1 - blend) + bottom_color[0] * blend)
    g = int(top_color[1] * (1 - blend) + bottom_color[1] * blend)
    b = int(top_color[2] * (1 - blend) + bottom_color[2] * blend)
    draw.line([(0, hero_top + i), (width, hero_top + i)], fill=(r, g, b))

# Subtle translucent dark band near bottom of hero to help white controls stand out
band_h = 56
band_y = hero_bottom - band_h
band_color = (20, 20, 20)
draw.rectangle([0, band_y, width, hero_bottom], fill=band_color)

# Thin progress-like separators over the hero area (background-only thin bars)
pb_y = hero_bottom - 36
draw.rectangle([48, pb_y, width - 48, pb_y + 6], fill=(230, 230, 230))
draw.rectangle([96, pb_y + 10, width - 96, pb_y + 14], fill=(200, 200, 200))

# Title/content area background remains the page bg. Add padding-top divider under hero
draw.line([(48, hero_bottom + 8), (width - 48, hero_bottom + 8)], fill=(245, 245, 247), width=2)

# Organizer card background (rounded rectangle) - do NOT draw icons/text inside it
org_card_x1 = 48
org_card_x2 = width - 48
org_card_y1 = 1080
org_card_y2 = 1248
org_card_radius = 28
org_bg = (247, 249, 251)
org_border = (232, 236, 241)

# subtle shadow (simulated by a slightly darker rounded rect offset)
shadow_offset = 6
draw.rounded_rectangle([org_card_x1, org_card_y1 + shadow_offset,
                        org_card_x2, org_card_y2 + shadow_offset],
                       radius=org_card_radius, fill=(238, 240, 243))
draw.rounded_rectangle([org_card_x1, org_card_y1, org_card_x2, org_card_y2],
                       radius=org_card_radius, fill=org_bg, outline=org_border, width=2)

# Small divider line inside content sections (e.g., between details and policy)
sep_y = 1560
draw.line([(48, sep_y), (width - 48, sep_y)], fill=(239, 239, 242), width=2)

# Location / info icon row separator region background (keep subtle)
info_region_y1 = 1280
info_region_y2 = 1560
draw.rectangle([48, info_region_y1, width - 48, info_region_y2], fill=bg_color)

# About this event header divider and spacing
about_y = 1860
draw.line([(48, about_y), (width - 48, about_y)], fill=(245, 245, 247), width=2)

# "About" section background area (white card-like region)
about_card_y1 = about_y + 24
about_card_y2 = about_card_y1 + 180
about_card_pad = 0
draw.rectangle([48 - about_card_pad, about_card_y1, width - 48 + about_card_pad, about_card_y2], fill=bg_color)

# Category pill placeholder background (rounded pill behind categories)
pill_x1 = 48
pill_x2 = 760
pill_y1 = about_card_y1 + 16
pill_y2 = pill_y1 + 48
draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2], radius=24, fill=(245, 247, 250), outline=None)

# Light horizontal rule below the short about text
draw.line([(48, about_card_y2 + 12), (width - 48, about_card_y2 + 12)], fill=(245, 245, 247), width=2)

# Ticket selection card (blue bordered rounded rectangle) placed above the reserve area
ticket_x1 = 48
ticket_x2 = width - 48
ticket_y1 = 2040
ticket_y2 = 2260
ticket_radius = 20
ticket_border_color = (58, 94, 255)  # blue border for ticket selection
ticket_bg = (255, 255, 255)

# Outer border
draw.rounded_rectangle([ticket_x1, ticket_y1, ticket_x2, ticket_y2],
                       radius=ticket_radius, outline=ticket_border_color, width=6, fill=ticket_bg)
# Inner pale background to simulate depth
inner_pad = 12
draw.rounded_rectangle([ticket_x1 + inner_pad, ticket_y1 + inner_pad,
                        ticket_x2 - inner_pad, ticket_y2 - inner_pad],
                       radius=ticket_radius - 6, fill=(250, 251, 253))

# Quantity control area background (subtle rounded rect on the right side inside ticket)
qty_w = 120
qty_x2 = ticket_x2 - 28
qty_x1 = qty_x2 - qty_w
qty_y1 = ticket_y1 + 36
qty_y2 = qty_y1 + 72
draw.rounded_rectangle([qty_x1, qty_y1, qty_x2, qty_y2], radius=14, fill=(245, 247, 250))

# Thin separator line above the ticket card (to indicate new section)
draw.line([(48, ticket_y1 - 20), (ticket_x2, ticket_y1 - 20)], fill=(235, 235, 238), width=1)

# Final subtle footer area divider (just above reserve button region)
reserve_region_y = 2324  # detected reserve region starts here; draw only a divider above it
draw.line([(0, reserve_region_y), (width, reserve_region_y)], fill=(238, 238, 240), width=2)

# End of background/structure drawing.
# (Actual icons, texts, and interactive buttons will be pasted on top by the caller.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1163), _c0)
except Exception:
    pass
layout["Following"] = [946, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 114, 107)
    canvas.paste(_c1, (987, 2439), _c1)
except Exception:
    pass
layout["icon_1"] = [987, 2439, 1101, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 109, 104)
    canvas.paste(_c2, (1214, 2441), _c2)
except Exception:
    pass
layout["icon_2"] = [1214, 2441, 1323, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1440, 636)
    canvas.paste(_c3, (0, 2324), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/05_icon_More.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/06_icon_9.13.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 108), _c6)
except Exception:
    pass
layout["9.13"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 92, 102)
    canvas.paste(_c7, (1108, 2442), _c7)
except Exception:
    pass
layout["icon_7"] = [1108, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 69)
    canvas.paste(_c8, (1155, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1155, 2, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 46, 63)
    canvas.paste(_c9, (1326, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1326, 3, 1372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 101, 64)
    canvas.paste(_c10, (1213, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1213, 2, 1314, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/11_icon_Business_Professional_._Startups_Small_B.png
try:
    _c11 = get_crop(11, 1440, 636)
    canvas.paste(_c11, (0, 2324), _c11)
except Exception:
    pass
layout["Business_&_Professional_."] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/12_icon_I_00_PM.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1116, 108), _c12)
except Exception:
    pass
layout["I:00_PM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/13_icon_130_Followers.png
try:
    _c13 = get_crop(13, 398, 144)
    canvas.paste(_c13, (946, 1163), _c13)
except Exception:
    pass
layout["130_Followers"] = [946, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/14_icon_9.13.png
try:
    _c14 = get_crop(14, 52, 62)
    canvas.paste(_c14, (117, 2), _c14)
except Exception:
    pass
layout["9.13"] = [117, 2, 169, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/15_icon_Free.png
try:
    _c15 = get_crop(15, 142, 116)
    canvas.paste(_c15, (94, 2567), _c15)
except Exception:
    pass
layout["Free"] = [94, 2567, 236, 2683]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/16_icon_9.13.png
try:
    _c16 = get_crop(16, 52, 59)
    canvas.paste(_c16, (183, 3), _c16)
except Exception:
    pass
layout["9.13"] = [183, 3, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/17_icon_Free.png
try:
    _c17 = get_crop(17, 104, 115)
    canvas.paste(_c17, (232, 2572), _c17)
except Exception:
    pass
layout["Free"] = [232, 2572, 336, 2687]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/18_text_9.13.png
try:
    _c18 = get_crop(18, 91, 43)
    canvas.paste(_c18, (20, 17), _c18)
except Exception:
    pass
layout["9.13"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/19_text_Saturday_April_13.png
try:
    _c19 = get_crop(19, 453, 77)
    canvas.paste(_c19, (38, 758), _c19)
except Exception:
    pass
layout["Saturday;_April_13"] = [38, 758, 491, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/20_text_I_00_PM.png
try:
    _c20 = get_crop(20, 207, 56)
    canvas.paste(_c20, (523, 766), _c20)
except Exception:
    pass
layout["I:00_PM"] = [523, 766, 730, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/21_text_Minorities_Building_Wealth_with.png
try:
    _c21 = get_crop(21, 269, 144)
    canvas.paste(_c21, (288, 1123), _c21)
except Exception:
    pass
layout["Minorities_Building_Wealt"] = [288, 1123, 557, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/22_text_At_the_Franchise_Expo_West.png
try:
    _c22 = get_crop(22, 398, 144)
    canvas.paste(_c22, (946, 1163), _c22)
except Exception:
    pass
layout["At_the_Franchise_Expo_Wes"] = [946, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/23_text_Los_Angeles_Convention_Center.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1390), _c23)
except Exception:
    pass
layout["Los_Angeles_Convention_Ce"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/24_text_hrs.png
try:
    _c24 = get_crop(24, 77, 50)
    canvas.paste(_c24, (176, 1547), _c24)
except Exception:
    pass
layout["hrs"] = [176, 1547, 253, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/25_text_Refund_policy.png
try:
    _c25 = get_crop(25, 299, 63)
    canvas.paste(_c25, (138, 1653), _c25)
except Exception:
    pass
layout["Refund_policy"] = [138, 1653, 437, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/26_text_The_organizer_will_review_refund_request.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1390), _c26)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/27_text_About_this_event.png
try:
    _c27 = get_crop(27, 452, 61)
    canvas.paste(_c27, (45, 1953), _c27)
except Exception:
    pass
layout["About_this_event"] = [45, 1953, 497, 2014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/28_text_We_invite_you_to_a_value-packed_educatio.png
try:
    _c28 = get_crop(28, 1440, 636)
    canvas.paste(_c28, (0, 2324), _c28)
except Exception:
    pass
layout["We_invite_you_to_a_value-"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/29_text_Complimentary_Access.png
try:
    _c29 = get_crop(29, 1440, 636)
    canvas.paste(_c29, (0, 2324), _c29)
except Exception:
    pass
layout["Complimentary_Access"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_12_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-14/30_clickable_Organizer_profile_picture.png
try:
    _c30 = get_crop(30, 144, 144)
    canvas.paste(_c30, (96, 1162), _c30)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1162, 240, 1306]
