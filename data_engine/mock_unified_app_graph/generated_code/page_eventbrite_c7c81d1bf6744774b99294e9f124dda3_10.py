# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_10
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12.png
# step_index: 10/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the mobile UI page (PIL draw commands)
# Uses existing variables: canvas (PIL Image), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

# Canvas size: 1440x2960

# Colors
status_bar_color = (238, 238, 241)    # light gray top status bar
status_divider = (210, 210, 215)
hero_bg = (36, 40, 46)                # dark hero/image background
card_bg = (247, 247, 251)             # very light card background
muted_divider = (236, 236, 240)
content_bg = (255, 255, 255)
footer_bg = (250, 250, 252)
soft_shadow = (225, 224, 230)

# 1) Status bar area (top)
status_height = 110
draw.rectangle([(0, 0), (1440, status_height)], fill=status_bar_color)
# subtle bottom divider under status bar
draw.line([(0, status_height), (1440, status_height)], fill=status_divider, width=1)

# 2) Hero / image background area (large dark area under status bar)
hero_top = status_height
hero_bottom = 520
draw.rectangle([(0, hero_top), (1440, hero_bottom)], fill=hero_bg)

# subtle horizontal indicator area (to mimic image carousel area background)
carousel_bar_y = hero_bottom - 40
draw.rectangle([(48, carousel_bar_y), (1392, carousel_bar_y + 6)], fill=(255,255,255,30))  # subtle bar-like strip

# thin divider below hero area
draw.line([(48, hero_bottom), (1392, hero_bottom)], fill=muted_divider, width=1)

# 3) Main content background (white)
content_top = hero_bottom + 24
draw.rectangle([(0, content_top), (1440, 2600)], fill=content_bg)

# 4) Organizer / profile card background (rounded rectangle)
org_card_left = 48
org_card_top = 1088
org_card_right = 1392
org_card_bottom = 1248
org_radius = 26
draw.rounded_rectangle([(org_card_left, org_card_top), (org_card_right, org_card_bottom)],
                       radius=org_radius, fill=card_bg, outline=None)

# subtle shadow/separator under the organizer card
draw.line([(org_card_left, org_card_bottom + 8), (org_card_right, org_card_bottom + 8)], fill=soft_shadow, width=1)

# 5) Divider line between info sections
divider_y = 1360
draw.line([(48, divider_y), (1392, divider_y)], fill=muted_divider, width=1)

# 6) Info section separators (for Masada / duration / refund rows)
row_start_x = 48
row_end_x = 1392
row_y_positions = [1320, 1408, 1500]  # approximate separators under each info block
for y in row_y_positions:
    draw.line([(row_start_x, y), (row_end_x, y)], fill=(245,245,247), width=1)

# 7) "Select date and time" section container background (subtle)
select_section_top = 1820
select_section_bottom = 2460
select_section_left = 24
select_section_right = 1416
draw.rectangle([(select_section_left, select_section_top), (select_section_right, select_section_bottom)], fill=content_bg)

# light top divider for the select section
draw.line([(select_section_left, select_section_top), (select_section_right, select_section_top)], fill=muted_divider, width=1)

# subtle card background band behind date-card region (but DO NOT draw individual date cards)
band_top = 1960
band_bottom = 2320
draw.rounded_rectangle([(36, band_top), (1404, band_bottom)], radius=14, fill=(255,255,255), outline=(245,245,247), width=2)

# 8) Horizontal separator above sticky footer
footer_top = 2680
draw.line([(0, footer_top), (1440, footer_top)], fill=muted_divider, width=1)

# 9) Sticky footer background (bottom bar)
draw.rectangle([(0, footer_top), (1440, 2960)], fill=footer_bg)

# subtle inner highlight on footer
draw.line([(24, footer_top + 6), (1416, footer_top + 6)], fill=(255,255,255), width=1)

# 10) Left price area background (within footer) - DO NOT draw text/button itself
price_area = (24, footer_top + 28, 660, 2960 - 28)
draw.rectangle([price_area[0:2], price_area[2:4]], fill=footer_bg)

# 11) Light outer margins / page edges subtle shadow
edge_shadow_width = 8
draw.rectangle([(0, 0), (edge_shadow_width, 2960)], fill=(250,250,250))
draw.rectangle([(1440-edge_shadow_width, 0), (1440, 2960)], fill=(250,250,250))

# Note: All actual icons, text and buttons will be pasted on top at their detected positions.
# This code strictly draws only background blocks, dividers, and structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1163), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/02_icon_27.png
try:
    _c2 = get_crop(2, 450, 516)
    canvas.paste(_c2, (24, 2067), _c2)
except Exception:
    pass
layout["27"] = [24, 2067, 474, 2583]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/03_icon_More.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/04_icon_7.10.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["7.10"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/05_icon_Share.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1260, 108), _c5)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/06_icon_May.png
try:
    _c6 = get_crop(6, 450, 516)
    canvas.paste(_c6, (474, 2067), _c6)
except Exception:
    pass
layout["May"] = [474, 2067, 924, 2583]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 70)
    canvas.paste(_c7, (1155, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1155, 1, 1203, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/08_icon_11.png
try:
    _c8 = get_crop(8, 450, 516)
    canvas.paste(_c8, (924, 2067), _c8)
except Exception:
    pass
layout["11"] = [924, 2067, 1374, 2583]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 44, 64)
    canvas.paste(_c9, (1328, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [1328, 2, 1372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/10_icon_ChicAfrik_FA_Events.png
try:
    _c10 = get_crop(10, 470, 144)
    canvas.paste(_c10, (288, 1123), _c10)
except Exception:
    pass
layout["ChicAfrik_&_FA_Events"] = [288, 1123, 758, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 96, 68)
    canvas.paste(_c11, (1214, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1214, 0, 1310, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/12_icon_7.10.png
try:
    _c12 = get_crop(12, 61, 70)
    canvas.paste(_c12, (181, 0), _c12)
except Exception:
    pass
layout["7.10"] = [181, 0, 242, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/13_icon_Masada.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (96, 1162), _c13)
except Exception:
    pass
layout["Masada"] = [96, 1162, 240, 1306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/14_text_7.10.png
try:
    _c14 = get_crop(14, 92, 41)
    canvas.paste(_c14, (22, 17), _c14)
except Exception:
    pass
layout["7.10"] = [22, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/15_text_Saturday_April_27.png
try:
    _c15 = get_crop(15, 449, 77)
    canvas.paste(_c15, (38, 758), _c15)
except Exception:
    pass
layout["Saturday,_April_27"] = [38, 758, 487, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/16_text_10_00_PM.png
try:
    _c16 = get_crop(16, 240, 62)
    canvas.paste(_c16, (523, 763), _c16)
except Exception:
    pass
layout["10:00_PM"] = [523, 763, 763, 825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/17_text_SATURDAYZ_MASADA_Afrobeats.png
try:
    _c17 = get_crop(17, 470, 144)
    canvas.paste(_c17, (288, 1123), _c17)
except Exception:
    pass
layout["SATURDAYZ_@_MASADA:_Afrob"] = [288, 1123, 758, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/18_text_Amapiano_Dancehall.png
try:
    _c18 = get_crop(18, 470, 144)
    canvas.paste(_c18, (288, 1123), _c18)
except Exception:
    pass
layout["Amapiano,_Dancehall"] = [288, 1123, 758, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/19_text_Masada.png
try:
    _c19 = get_crop(19, 177, 52)
    canvas.paste(_c19, (141, 1438), _c19)
except Exception:
    pass
layout["Masada"] = [141, 1438, 318, 1490]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/20_text_5_hrs.png
try:
    _c20 = get_crop(20, 112, 50)
    canvas.paste(_c20, (141, 1547), _c20)
except Exception:
    pass
layout["5_hrs"] = [141, 1547, 253, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/21_text_Refund_policy.png
try:
    _c21 = get_crop(21, 299, 63)
    canvas.paste(_c21, (138, 1653), _c21)
except Exception:
    pass
layout["Refund_policy"] = [138, 1653, 437, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/22_text_The_organizer_will_review_refund_request.png
try:
    _c22 = get_crop(22, 1344, 144)
    canvas.paste(_c22, (48, 1390), _c22)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/23_text_Select_date_and_time.png
try:
    _c23 = get_crop(23, 450, 516)
    canvas.paste(_c23, (24, 2067), _c23)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 2067, 474, 2583]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/24_text_Saturday.png
try:
    _c24 = get_crop(24, 188, 57)
    canvas.paste(_c24, (1054, 2142), _c24)
except Exception:
    pass
layout["Saturday"] = [1054, 2142, 1242, 2199]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/25_text_SO_-_S200.png
try:
    _c25 = get_crop(25, 228, 61)
    canvas.paste(_c25, (89, 2811), _c25)
except Exception:
    pass
layout["SO_-_S200"] = [89, 2811, 317, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_10_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-12/26_text_May.png
try:
    _c26 = get_crop(26, 93, 57)
    canvas.paste(_c26, (1103, 2225), _c26)
except Exception:
    pass
layout["May"] = [1103, 2225, 1196, 2282]
