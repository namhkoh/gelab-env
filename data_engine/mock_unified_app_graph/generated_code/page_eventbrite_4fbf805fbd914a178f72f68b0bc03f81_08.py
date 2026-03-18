# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_08
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10.png
# step_index: 8/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the Event page
# Assumes existing variables: canvas (PIL Image 1440x2960 RGB) and draw (ImageDraw)
# and fonts: font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
status_bar_color = (48, 48, 48)        # dark status bar
status_border = (70, 70, 70)
banner_color = (8, 8, 8)               # black banner behind header image
content_bg = (255, 255, 255)           # page background (white)
card_bg = (246, 247, 250)              # light card background
card_border = (226, 227, 232)          # subtle border for cards
muted_line = (235, 235, 238)           # separators
accent_blue = (60, 84, 255)            # used subtly for selection outlines
muted_purple = (99, 57, 118)          # deep purple for separators (soft)
light_gray = (243, 244, 246)

# 1) Status bar at top (~56px)
status_h = 56
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)
# thin bottom border for status bar
draw.line([(0, status_h-1), (W, status_h-1)], fill=status_border, width=1)

# 2) Header / hero banner background (image area)
banner_top = status_h
banner_bottom = 360
draw.rectangle([(0, banner_top), (W, banner_bottom)], fill=banner_color)

# subtle top shadow at junction of banner -> content
draw.rectangle([(0, banner_bottom - 6), (W, banner_bottom)], fill=(0,0,0,0))

# 3) Main content area background (white) - explicit to ensure clean area
content_top = banner_bottom
draw.rectangle([(0, content_top), (W, H)], fill=content_bg)

# 4) Large event title area top spacing (no text drawn)
# Add subtle divider under title area to separate header from content
title_divider_y = content_top + 420
draw.line([(48, title_divider_y), (W-48, title_divider_y)], fill=muted_line, width=1)

# 5) Host / organizer card (rounded light card)
host_card_x1 = 48
host_card_x2 = W - 48
host_card_y1 = content_top + 180
host_card_y2 = host_card_y1 + 150
draw.rounded_rectangle(
    [(host_card_x1, host_card_y1), (host_card_x2, host_card_y2)],
    radius=20,
    fill=card_bg,
    outline=card_border,
    width=2
)

# 6) Small divider under host card
draw.line([(48, host_card_y2 + 28), (W-48, host_card_y2 + 28)], fill=muted_line, width=1)

# 7) Event metadata area (icons + lines) - draw subtle rows and separators (no icons/text)
meta_start_y = host_card_y2 + 56
row_height = 86
for i in range(3):
    y1 = meta_start_y + i*row_height
    y2 = y1 + row_height
    # draw subtle background band for even rows to aid separation
    if i % 2 == 1:
        draw.rectangle([(48, y1), (W-48, y2)], fill=light_gray)
    # horizontal separator
    draw.line([(48, y2), (W-48, y2)], fill=muted_line, width=1)

# 8) Thin rule before "Select date and time" section
select_section_y = meta_start_y + 3*row_height + 24
draw.line([(48, select_section_y), (W-48, select_section_y)], fill=muted_line, width=1)

# 9) "Select date and time" card row placeholders (rounded date pills)
pills_top = select_section_y + 36
pill_h = 220
pill_w = 360
pill_gap = 36
pill_x = 48
# draw three pill cards across with subtle borders (no text inside)
for i in range(3):
    x1 = pill_x + i*(pill_w + pill_gap)
    x2 = x1 + pill_w
    y1 = pills_top
    y2 = y1 + pill_h
    # main pill background
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=22, fill=(255,255,255), outline=card_border, width=2)
    # inner subtle top line to mimic card elevation
    draw.line([(x1+12, y1+64), (x2-12, y1+64)], fill=muted_line, width=1)

# 10) Light separator area below pills
pills_bottom = pills_top + pill_h
draw.line([(48, pills_bottom + 28), (W-48, pills_bottom + 28)], fill=muted_line, width=1)

# 11) Content card area (placeholder) for additional details - subtle rounded rectangle
details_card_y1 = pills_bottom + 48
details_card_y2 = details_card_y1 + 320
draw.rounded_rectangle(
    [(48, details_card_y1), (W-48, details_card_y2)],
    radius=18,
    fill=(255,255,255),
    outline=card_border,
    width=1
)

# 12) Ensure we do NOT draw anything in the bottom reserve overlay area.
# The reserved seating / reserve button region starts at y = 2321 (per provided detection).
# Add a faint top shadow above that area so pasted overlay sits naturally.
reserve_area_top = 2321
# Draw a faint shadow line right above reserve area
draw.line([(0, reserve_area_top-6), (W, reserve_area_top-6)], fill=(220,220,220), width=2)
# And a very subtle fade band to visually separate content from overlay (won't overlap)
fade_band_top = reserve_area_top - 40
fade_band_bottom = reserve_area_top - 1
draw.rectangle([(0, fade_band_top), (W, fade_band_bottom)], fill=(250,250,250))

# 13) Final subtle vertical padding lines/margins to frame the content area
left_margin_x = 36
right_margin_x = W - 36
draw.line([(left_margin_x, 0), (left_margin_x, reserve_area_top-48)], fill=(248,248,249), width=1)
draw.line([(right_margin_x, 0), (right_margin_x, reserve_area_top-48)], fill=(248,248,249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1194), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1194, 1344, 1338]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/01_icon_Anril.png
try:
    _c1 = get_crop(1, 450, 193)
    canvas.paste(_c1, (24, 2128), _c1)
except Exception:
    pass
layout["Anril"] = [24, 2128, 474, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/02_icon_Anril.png
try:
    _c2 = get_crop(2, 450, 193)
    canvas.paste(_c2, (924, 2128), _c2)
except Exception:
    pass
layout["Anril"] = [924, 2128, 1374, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/03_icon_Anril.png
try:
    _c3 = get_crop(3, 450, 193)
    canvas.paste(_c3, (474, 2128), _c3)
except Exception:
    pass
layout["Anril"] = [474, 2128, 924, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/04_icon_Anril.png
try:
    _c4 = get_crop(4, 110, 103)
    canvas.paste(_c4, (988, 2437), _c4)
except Exception:
    pass
layout["Anril"] = [988, 2437, 1098, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/05_icon_Anril.png
try:
    _c5 = get_crop(5, 108, 101)
    canvas.paste(_c5, (1215, 2439), _c5)
except Exception:
    pass
layout["Anril"] = [1215, 2439, 1323, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/06_icon_The_SYSTEM.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1116, 108), _c6)
except Exception:
    pass
layout["The_SYSTEM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/07_icon_Anril.png
try:
    _c7 = get_crop(7, 90, 101)
    canvas.paste(_c7, (1109, 2439), _c7)
except Exception:
    pass
layout["Anril"] = [1109, 2439, 1199, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/08_icon_Reserve_a_spot.png
try:
    _c8 = get_crop(8, 1440, 639)
    canvas.paste(_c8, (0, 2321), _c8)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2321, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/09_icon_The_SYSTEM.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1260, 108), _c9)
except Exception:
    pass
layout["The_SYSTEM"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/10_icon_4.56.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (36, 108), _c10)
except Exception:
    pass
layout["4.56"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/11_icon_D_E74.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1116, 108), _c11)
except Exception:
    pass
layout["D,E74"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 43, 57)
    canvas.paste(_c12, (1327, 6), _c12)
except Exception:
    pass
layout["icon_12"] = [1327, 6, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/13_icon_Lora_Brown_owner_of_LKB.png
try:
    _c13 = get_crop(13, 773, 144)
    canvas.paste(_c13, (144, 1153), _c13)
except Exception:
    pass
layout["Lora_Brown,_owner_of_LKB"] = [144, 1153, 917, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 56)
    canvas.paste(_c14, (314, 5), _c14)
except Exception:
    pass
layout["icon_14"] = [314, 5, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 91, 55)
    canvas.paste(_c15, (1218, 6), _c15)
except Exception:
    pass
layout["icon_15"] = [1218, 6, 1309, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/16_icon_4.56.png
try:
    _c16 = get_crop(16, 56, 62)
    canvas.paste(_c16, (184, 2), _c16)
except Exception:
    pass
layout["4.56"] = [184, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 58)
    canvas.paste(_c17, (249, 4), _c17)
except Exception:
    pass
layout["icon_17"] = [249, 4, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/18_icon_Free.png
try:
    _c18 = get_crop(18, 104, 109)
    canvas.paste(_c18, (233, 2572), _c18)
except Exception:
    pass
layout["Free"] = [233, 2572, 337, 2681]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/19_icon_4.56.png
try:
    _c19 = get_crop(19, 56, 64)
    canvas.paste(_c19, (117, 1), _c19)
except Exception:
    pass
layout["4.56"] = [117, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 50, 59)
    canvas.paste(_c20, (382, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 3, 432, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/21_text_4.56.png
try:
    _c21 = get_crop(21, 92, 41)
    canvas.paste(_c21, (22, 17), _c21)
except Exception:
    pass
layout["4.56"] = [22, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/22_text_Wednesday_April_24.png
try:
    _c22 = get_crop(22, 515, 79)
    canvas.paste(_c22, (43, 756), _c22)
except Exception:
    pass
layout["Wednesday;_April_24"] = [43, 756, 558, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/23_text_6_00_PM.png
try:
    _c23 = get_crop(23, 209, 56)
    canvas.paste(_c23, (583, 766), _c23)
except Exception:
    pass
layout["6:00_PM"] = [583, 766, 792, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/24_text_The_Path_to_Wealth_Through_Education_-.png
try:
    _c24 = get_crop(24, 773, 144)
    canvas.paste(_c24, (144, 1153), _c24)
except Exception:
    pass
layout["The_Path_to_Wealth_Throug"] = [144, 1153, 917, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/25_text_Pacoima.png
try:
    _c25 = get_crop(25, 301, 72)
    canvas.paste(_c25, (44, 986), _c25)
except Exception:
    pass
layout["Pacoima"] = [44, 986, 345, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/26_text_ONLINE_Event.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1451), _c26)
except Exception:
    pass
layout["ONLINE_Event"] = [48, 1451, 1392, 1595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/27_text_Introduction_To_Our_Nationwide_Community.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1451), _c27)
except Exception:
    pass
layout["Introduction_To_Our_Natio"] = [48, 1451, 1392, 1595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/28_text_2_hrs.png
try:
    _c28 = get_crop(28, 112, 49)
    canvas.paste(_c28, (141, 1610), _c28)
except Exception:
    pass
layout["2_hrs"] = [141, 1610, 253, 1659]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/29_text_Refund_policy.png
try:
    _c29 = get_crop(29, 299, 63)
    canvas.paste(_c29, (138, 1713), _c29)
except Exception:
    pass
layout["Refund_policy"] = [138, 1713, 437, 1776]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/30_text_The_organizer_will_review_refund_request.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 1451), _c30)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1451, 1392, 1595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/31_text_Select_date_and_time.png
try:
    _c31 = get_crop(31, 450, 193)
    canvas.paste(_c31, (24, 2128), _c31)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 2128, 474, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/32_text_Reserved_Seating.png
try:
    _c32 = get_crop(32, 450, 193)
    canvas.paste(_c32, (24, 2128), _c32)
except Exception:
    pass
layout["Reserved_Seating"] = [24, 2128, 474, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/33_text_By_Invitation_Only.png
try:
    _c33 = get_crop(33, 1440, 639)
    canvas.paste(_c33, (0, 2321), _c33)
except Exception:
    pass
layout["By_Invitation_Only"] = [0, 2321, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_08_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-10/34_text_Free.png
try:
    _c34 = get_crop(34, 105, 49)
    canvas.paste(_c34, (116, 2595), _c34)
except Exception:
    pass
layout["Free"] = [116, 2595, 221, 2644]
