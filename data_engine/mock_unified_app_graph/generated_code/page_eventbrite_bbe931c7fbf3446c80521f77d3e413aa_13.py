# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_13
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15.png
# step_index: 13/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the UI page (uses provided canvas and draw)

# Colors
bg_color = (250, 250, 252)         # very light off-white background
status_bar_color = (200, 200, 200) # light grey status bar
divider_color = (235, 233, 239)    # subtle light divider
header_shadow = (230, 228, 235)    # shadow under header
card_border = (60, 90, 255)        # blue-ish card border for ticket box
card_bg = (255, 255, 255)          # card interior white
pill_bg = (244, 246, 251)          # pale pill background for category tag
soft_grey = (245, 245, 247)        # soft background for section blocks

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (top ~64 px)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)
# thin bottom line for status bar
draw.line([(0, status_h), (w, status_h)], fill=divider_color, width=1)

# Header / toolbar area under status bar (~96..220)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (w, header_bottom)], fill=(255, 255, 255))
# subtle shadow under header
draw.rectangle([(0, header_bottom-3), (w, header_bottom)], fill=header_shadow)

# Primary content divider after top header cluster (approx where refund policy section divider is)
divider1_y = 420
draw.line([(40, divider1_y), (w-40, divider1_y)], fill=divider_color, width=2)

# "About this event" title area is white; draw subtle background pill for category tag below it
# Pill location (approx under title)
pill_x0 = 48
pill_y0 = 760
pill_x1 = 640
pill_y1 = 820
draw.rounded_rectangle([(pill_x0, pill_y0), (pill_x1, pill_y1)], radius=36, fill=pill_bg, outline=None)

# Divider under About section (after description area)
divider2_y = 980
draw.line([(40, divider2_y), (w-40, divider2_y)], fill=divider_color, width=2)

# Location section divider under location details
divider3_y = 1180
draw.line([(40, divider3_y), (w-40, divider3_y)], fill=divider_color, width=2)

# Light background band behind location block to separate visually (subtle)
loc_band_top = 1100
loc_band_bottom = 1240
draw.rectangle([(0, loc_band_top), (w, loc_band_bottom)], fill=bg_color)

# Thin divider above organizer area
divider4_y = 1550
draw.line([(40, divider4_y), (w-40, divider4_y)], fill=divider_color, width=1)

# Organizer area background: a soft card effect centered around where avatar and organizer name sit
org_card_top = 1560
org_card_bottom = 1960
org_card_margin = 80
draw.rectangle([(org_card_margin, org_card_top), (w-org_card_margin, org_card_bottom)], fill=(255,255,255))
# very faint border around organizer card area
draw.rounded_rectangle([(org_card_margin, org_card_top), (w-org_card_margin, org_card_bottom)], radius=12, outline=(246,244,247), width=1)

# Ticket selection card (rounded rectangle with colored border)
ticket_card_top = 2280
ticket_card_bottom = 2520
ticket_card_margin_x = 60
ticket_card_rect = [(ticket_card_margin_x, ticket_card_top), (w-ticket_card_margin_x, ticket_card_bottom)]
# outer border
draw.rounded_rectangle(ticket_card_rect, radius=18, outline=card_border, width=6, fill=card_bg)
# inner padding area (slightly inset to create a subtle inner area)
inner_pad = 12
draw.rounded_rectangle([(ticket_card_margin_x+inner_pad, ticket_card_top+inner_pad),
                        (w-ticket_card_margin_x-inner_pad, ticket_card_bottom-inner_pad)],
                       radius=14, fill=card_bg, outline=None)

# Small info row background above ticket card (where "Free" label sits) - keep subtle
info_row_top = ticket_card_top - 120
info_row_bottom = ticket_card_top - 40
info_row_margin = 48
draw.rectangle([(info_row_margin, info_row_top), (w-info_row_margin, info_row_bottom)], fill=soft_grey, outline=None)
# divider under this info row
draw.line([(info_row_margin, info_row_bottom+6), (w-info_row_margin, info_row_bottom+6)], fill=divider_color, width=1)

# Reserve button area (do not draw the button itself - just a subtle separator above it)
reserve_sep_y = 2720
draw.line([(40, reserve_sep_y), (w-40, reserve_sep_y)], fill=divider_color, width=1)

# Bottom safe area (keep white)
bottom_safe_top = reserve_sep_y + 6
draw.rectangle([(0, bottom_safe_top), (w, h)], fill=bg_color)

# Additional subtle vertical margins/lines to match layout rhythm (not icons/text)
# Left content guide line (for alignment, very faint)
draw.line([(48, header_bottom+8), (48, h-200)], fill=(245,245,247), width=1)
# Right content guide line (faint)
draw.line([(w-48, header_bottom+8), (w-48, h-200)], fill=(245,245,247), width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/02_icon_Organized_by.png
try:
    _c2 = get_crop(2, 240, 240)
    canvas.paste(_c2, (600, 1846), _c2)
except Exception:
    pass
layout["Organized_by"] = [600, 1846, 840, 2086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/03_icon_9.13.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["9.13"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/04_icon_hrs.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["hrs"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/05_icon_Reserve_a_spot.png
try:
    _c5 = get_crop(5, 1296, 132)
    canvas.paste(_c5, (72, 2756), _c5)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/06_icon_Decrease.png
try:
    _c6 = get_crop(6, 99, 96)
    canvas.paste(_c6, (996, 2444), _c6)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/07_icon_Increase.png
try:
    _c7 = get_crop(7, 96, 96)
    canvas.paste(_c7, (1224, 2444), _c7)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 94, 102)
    canvas.paste(_c8, (1107, 2442), _c8)
except Exception:
    pass
layout["icon_8"] = [1107, 2442, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/09_icon_Free.png
try:
    _c9 = get_crop(9, 134, 100)
    canvas.paste(_c9, (100, 2576), _c9)
except Exception:
    pass
layout["Free"] = [100, 2576, 234, 2676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 56)
    canvas.paste(_c10, (316, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [316, 6, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 56)
    canvas.paste(_c11, (249, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [249, 5, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 43, 64)
    canvas.paste(_c12, (1157, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1157, 2, 1200, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 97, 60)
    canvas.paste(_c13, (1215, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1215, 3, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 56, 56)
    canvas.paste(_c14, (180, 5), _c14)
except Exception:
    pass
layout["icon_14"] = [180, 5, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 46, 56)
    canvas.paste(_c15, (1325, 5), _c15)
except Exception:
    pass
layout["icon_15"] = [1325, 5, 1371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/16_icon_Free.png
try:
    _c16 = get_crop(16, 75, 72)
    canvas.paste(_c16, (249, 2588), _c16)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/17_icon_9.13.png
try:
    _c17 = get_crop(17, 54, 58)
    canvas.paste(_c17, (116, 4), _c17)
except Exception:
    pass
layout["9.13"] = [116, 4, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/18_icon_Show_map.png
try:
    _c18 = get_crop(18, 226, 144)
    canvas.paste(_c18, (1166, 1246), _c18)
except Exception:
    pass
layout["Show_map"] = [1166, 1246, 1392, 1390]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/19_icon_Refund_policy.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["Refund_policy"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/20_icon_Los_Angeles_Convention_Center_1201_South.png
try:
    _c20 = get_crop(20, 240, 240)
    canvas.paste(_c20, (600, 1846), _c20)
except Exception:
    pass
layout["Los_Angeles_Convention_Ce"] = [600, 1846, 840, 2086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 43, 57)
    canvas.paste(_c21, (386, 5), _c21)
except Exception:
    pass
layout["icon_21"] = [386, 5, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/22_text_9.13.png
try:
    _c22 = get_crop(22, 91, 43)
    canvas.paste(_c22, (20, 17), _c22)
except Exception:
    pass
layout["9.13"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/23_text_Minorities_Building_..png
try:
    _c23 = get_crop(23, 556, 79)
    canvas.paste(_c23, (250, 150), _c23)
except Exception:
    pass
layout["Minorities_Building_."] = [250, 150, 806, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/24_text_The_organizer_will_review_refund_request.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1116, 108), _c24)
except Exception:
    pass
layout["The_organizer_will_review"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/25_text_About_this_event.png
try:
    _c25 = get_crop(25, 450, 57)
    canvas.paste(_c25, (46, 682), _c25)
except Exception:
    pass
layout["About_this_event"] = [46, 682, 496, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/26_text_Business_Professional_._Startups_Small_B.png
try:
    _c26 = get_crop(26, 234, 144)
    canvas.paste(_c26, (48, 1028), _c26)
except Exception:
    pass
layout["Business_&_Professional_."] = [48, 1028, 282, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/27_text_We_invite_you_to_a_value-packed_educatio.png
try:
    _c27 = get_crop(27, 234, 144)
    canvas.paste(_c27, (48, 1028), _c27)
except Exception:
    pass
layout["We_invite_you_to_a_value-"] = [48, 1028, 282, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/28_text_Read_more.png
try:
    _c28 = get_crop(28, 234, 144)
    canvas.paste(_c28, (48, 1028), _c28)
except Exception:
    pass
layout["Read_more"] = [48, 1028, 282, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/29_text_Location.png
try:
    _c29 = get_crop(29, 244, 61)
    canvas.paste(_c29, (43, 1292), _c29)
except Exception:
    pass
layout["Location"] = [43, 1292, 287, 1353]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/30_text_Organized_by.png
try:
    _c30 = get_crop(30, 341, 119)
    canvas.paste(_c30, (550, 2205), _c30)
except Exception:
    pass
layout["Organized_by"] = [550, 2205, 891, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/31_text_Sociallybuzz.png
try:
    _c31 = get_crop(31, 341, 119)
    canvas.paste(_c31, (550, 2205), _c31)
except Exception:
    pass
layout["Sociallybuzz"] = [550, 2205, 891, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_13_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-15/32_text_Complimentary_Access.png
try:
    _c32 = get_crop(32, 75, 72)
    canvas.paste(_c32, (249, 2588), _c32)
except Exception:
    pass
layout["Complimentary_Access"] = [249, 2588, 324, 2660]
