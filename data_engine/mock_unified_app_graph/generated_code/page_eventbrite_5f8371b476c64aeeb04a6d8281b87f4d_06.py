# page_id: page_eventbrite_5f8371b476c64aeeb04a6d8281b87f4d_06
# screenshot: 2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8.png
# step_index: 6/7
# task: Open Eventbrite. Search Science & Tech event. Select the first one that is not promoted. If it is free, add it to Favorites. If it is not free, record its price in Google Keep Notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint backgrounds and structure for the mobile UI
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_color = (255, 255, 255)            # overall page background (white)
status_bar_color = (211, 211, 211)    # light gray status bar
banner_start = (79, 189, 201)         # teal gradient start
banner_end = (45, 120, 141)           # teal gradient end
card_bg = (247, 246, 249)             # very light card background
muted_divider = (230, 228, 235)       # subtle divider
ticket_border = (47, 86, 240)         # blue border for ticket card
shadow_color = (220, 220, 220)        # light shadow

# Fill canvas (ensure clean base)
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar (top ~50px)
status_h = 86
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Top banner / hero image area (approx)
banner_y0 = status_h
banner_y1 = 460
# Draw a vertical gradient for the banner
steps = 60
for i in range(steps):
    t = i / max(1, steps - 1)
    r = int(banner_start[0] * (1 - t) + banner_end[0] * t)
    g = int(banner_start[1] * (1 - t) + banner_end[1] * t)
    b = int(banner_start[2] * (1 - t) + banner_end[2] * t)
    y0 = int(banner_y0 + (banner_y1 - banner_y0) * (i / steps))
    y1 = int(banner_y0 + (banner_y1 - banner_y0) * ((i + 1) / steps))
    draw.rectangle([(0, y0), (W, y1)], fill=(r, g, b))

# Soft dark overlay band at bottom of banner to emulate photo fade
overlay_h = 46
draw.rectangle([(0, banner_y1 - overlay_h), (W, banner_y1)], fill=(0, 0, 0, 30))

# Thin divider under banner
draw.rectangle([(48, banner_y1 + 12), (W - 48, banner_y1 + 14)], fill=muted_divider)

# Organizer card (rounded rectangle) under title area
card_x0 = 48
card_x1 = W - 48
card_y0 = 1150
card_y1 = 1300
# shadow
draw.rounded_rectangle([(card_x0 + 0, card_y0 + 6), (card_x1 + 0, card_y1 + 12)],
                       radius=20, fill=shadow_color)
# card body
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)],
                       radius=20, fill=card_bg)

# Subtle divider line beneath details section
divider_y = 1650
draw.rectangle([(48, divider_y), (W - 48, divider_y + 2)], fill=muted_divider)

# "About this event" separator area - label area left clear for text/icons
about_y = 1880
draw.rectangle([(48, about_y), (W - 48, about_y + 6)], fill=(255, 255, 255))

# Another muted divider before tickets
divider2_y = 2040
draw.rectangle([(48, divider2_y), (W - 48, divider2_y + 2)], fill=muted_divider)

# Ticket selection card (blue bordered rounded rectangle) - keep above reserve area
ticket_x0 = 48
ticket_x1 = W - 48
ticket_y0 = 2100
ticket_y1 = 2276  # must remain above reserve region (reserve starts at y=2324)
# subtle shadow under ticket card
draw.rounded_rectangle([(ticket_x0 + 0, ticket_y0 + 6), (ticket_x1 + 0, ticket_y1 + 12)],
                       radius=18, fill=shadow_color)
# white card interior
draw.rounded_rectangle([(ticket_x0, ticket_y0), (ticket_x1, ticket_y1)],
                       radius=18, fill=(255, 255, 255), outline=ticket_border, width=6)

# Small inner separator inside the ticket card (left area)
inner_sep_x = ticket_x0 + 30
draw.rectangle([(inner_sep_x, ticket_y0 + 82), (ticket_x1 - 30, ticket_y0 + 84)], fill=(245, 245, 250))

# Top-level horizontal separator just above the reserved bottom area
reserve_start_y = 2324
draw.rectangle([(0, reserve_start_y - 6), (W, reserve_start_y - 4)], fill=muted_divider)

# Add faint horizontal section separators for visual structure
sep_positions = [banner_y1 + 110, card_y1 + 80, divider_y + 220, divider2_y + 120]
for y in sep_positions:
    draw.rectangle([(48, y), (W - 48, y + 1)], fill=(240, 239, 243))

# Decorative rounded corner accents (purely background)
accent_radius = 28
draw.ellipse([(-accent_radius, banner_y1 - 2 * accent_radius),
              (accent_radius, banner_y1)], fill=banner_end)
draw.ellipse([(W - accent_radius, banner_y1 - 2 * accent_radius),
              (W + accent_radius, banner_y1)], fill=banner_end)

# End of structural/background drawing
# (No text, icons, or buttons are drawn — those will be pasted later)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/01_icon_Going_fast.png
try:
    _c1 = get_crop(1, 335, 86)
    canvas.paste(_c1, (41, 753), _c1)
except Exception:
    pass
layout["Going_fast"] = [41, 753, 376, 839]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 113, 106)
    canvas.paste(_c2, (987, 2439), _c2)
except Exception:
    pass
layout["icon_2"] = [987, 2439, 1100, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1440, 636)
    canvas.paste(_c3, (0, 2324), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/04_icon_More.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1116, 108), _c4)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 108, 105)
    canvas.paste(_c5, (1214, 2441), _c5)
except Exception:
    pass
layout["icon_5"] = [1214, 2441, 1322, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 93, 102)
    canvas.paste(_c6, (1107, 2442), _c6)
except Exception:
    pass
layout["icon_6"] = [1107, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/07_icon_Science_Technology_._High_Tech.png
try:
    _c7 = get_crop(7, 718, 100)
    canvas.paste(_c7, (34, 2071), _c7)
except Exception:
    pass
layout["Science_&_Technology_._Hi"] = [34, 2071, 752, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/08_icon_9.38.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["9.38"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/09_icon_Share.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1260, 108), _c9)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 43, 57)
    canvas.paste(_c10, (1327, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [1327, 5, 1370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 40, 44)
    canvas.paste(_c11, (237, 596), _c11)
except Exception:
    pass
layout["icon_11"] = [237, 596, 277, 640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 33, 43)
    canvas.paste(_c12, (279, 596), _c12)
except Exception:
    pass
layout["icon_12"] = [279, 596, 312, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 35, 45)
    canvas.paste(_c13, (197, 596), _c13)
except Exception:
    pass
layout["icon_13"] = [197, 596, 232, 641]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/14_icon_Register_for_Break_Into_Tech_nowl.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1116, 108), _c14)
except Exception:
    pass
layout["Register_for_Break_Into_T"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 47, 55)
    canvas.paste(_c15, (1267, 5), _c15)
except Exception:
    pass
layout["icon_15"] = [1267, 5, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 56, 56)
    canvas.paste(_c16, (1218, 5), _c16)
except Exception:
    pass
layout["icon_16"] = [1218, 5, 1274, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 39, 45)
    canvas.paste(_c17, (315, 595), _c17)
except Exception:
    pass
layout["icon_17"] = [315, 595, 354, 640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 49, 55)
    canvas.paste(_c18, (185, 6), _c18)
except Exception:
    pass
layout["icon_18"] = [185, 6, 234, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/19_icon_hrs_30_mins.png
try:
    _c19 = get_crop(19, 290, 75)
    canvas.paste(_c19, (140, 1567), _c19)
except Exception:
    pass
layout["hrs_30_mins"] = [140, 1567, 430, 1642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/20_icon_9.38.png
try:
    _c20 = get_crop(20, 51, 60)
    canvas.paste(_c20, (118, 3), _c20)
except Exception:
    pass
layout["9.38"] = [118, 3, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/21_text_9.38.png
try:
    _c21 = get_crop(21, 94, 43)
    canvas.paste(_c21, (20, 17), _c21)
except Exception:
    pass
layout["9.38"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/22_text_Tuesday_March_26.png
try:
    _c22 = get_crop(22, 516, 144)
    canvas.paste(_c22, (144, 1155), _c22)
except Exception:
    pass
layout["Tuesday;_March_26"] = [144, 1155, 660, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/23_text_6.00_PM.png
try:
    _c23 = get_crop(23, 213, 63)
    canvas.paste(_c23, (541, 890), _c23)
except Exception:
    pass
layout["6.00_PM"] = [541, 890, 754, 953]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/24_text_Break_into_Tech_New_York.png
try:
    _c24 = get_crop(24, 516, 144)
    canvas.paste(_c24, (144, 1155), _c24)
except Exception:
    pass
layout["Break_into_Tech:_New_York"] = [144, 1155, 660, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/25_text_Northeastern_University.png
try:
    _c25 = get_crop(25, 516, 144)
    canvas.paste(_c25, (144, 1155), _c25)
except Exception:
    pass
layout["Northeastern_University"] = [144, 1155, 660, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/26_text_207_Followers.png
try:
    _c26 = get_crop(26, 516, 144)
    canvas.paste(_c26, (144, 1155), _c26)
except Exception:
    pass
layout["207_Followers"] = [144, 1155, 660, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/27_text_Suites_Fifth_Avenue.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1422), _c27)
except Exception:
    pass
layout["Suites_Fifth_Avenue"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/28_text_Refund_policy.png
try:
    _c28 = get_crop(28, 299, 63)
    canvas.paste(_c28, (138, 1685), _c28)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/29_text_The_organizer_will_review_refund_request.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1422), _c29)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/30_text_About_this_event.png
try:
    _c30 = get_crop(30, 453, 67)
    canvas.paste(_c30, (44, 1982), _c30)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 497, 2049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/31_text_Join_our_Break_into_Tech_event_to_discov.png
try:
    _c31 = get_crop(31, 1440, 636)
    canvas.paste(_c31, (0, 2324), _c31)
except Exception:
    pass
layout["Join_our_Break_into_Tech_"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/32_text_BiT_Boston.png
try:
    _c32 = get_crop(32, 244, 49)
    canvas.paste(_c32, (116, 2454), _c32)
except Exception:
    pass
layout["BiT_Boston"] = [116, 2454, 360, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_06_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-8/33_text_Free.png
try:
    _c33 = get_crop(33, 105, 48)
    canvas.paste(_c33, (116, 2599), _c33)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]
