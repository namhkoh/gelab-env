# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_07
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9.png
# step_index: 7/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 64
draw.rectangle([(0, 0), (1440, status_h)], fill="#d0d0d0")

# Header image/banner area (subtle vertical gradient placeholder)
hdr_top = status_h
hdr_bottom = 520
for i in range(hdr_top, hdr_bottom):
    # gradient from light gray to near-white
    t = (i - hdr_top) / max(1, (hdr_bottom - hdr_top))
    r = int(240 + (255 - 240) * t)
    g = int(240 + (255 - 240) * t)
    b = int(244 + (255 - 244) * t)
    draw.line([(0, i), (1440, i)], fill=(r, g, b))

# Soft overlay at bottom of header to hint image fade
overlay_height = 60
for i in range(overlay_height):
    alpha = int(200 * (1 - i / overlay_height))
    y = hdr_bottom - overlay_height + i
    # blend with white by drawing a semi-transparent-looking line (simulated)
    r = int(255 - alpha * 0.08)
    g = int(255 - alpha * 0.08)
    b = int(255 - alpha * 0.06)
    draw.line([(0, y), (1440, y)], fill=(r, g, b))

# Thin divider under header
draw.line([(40, hdr_bottom + 4), (1400, hdr_bottom + 4)], fill="#e6e6e6", width=2)

# Organizer card (rounded light card with subtle border and shadow)
org_x1, org_x2 = 48, 1392
org_top = 1140
org_bottom = 1275
shadow_offset = 8
# shadow
draw.rounded_rectangle(
    [(org_x1 + shadow_offset, org_top + shadow_offset), (org_x2 + shadow_offset, org_bottom + shadow_offset)],
    radius=20,
    fill="#f2f0f5"
)
# card
draw.rounded_rectangle(
    [(org_x1, org_top), (org_x2, org_bottom)],
    radius=20,
    fill="#faf9fb",
    outline="#e6e1ee",
    width=2
)

# Small subtle divider under organizer section
sep_y = org_bottom + 70
draw.line([(48, sep_y), (1392, sep_y)], fill="#f0eef2", width=2)

# Info section background block (light to contain location/time/refund info)
info_x1, info_x2 = 40, 1400
info_top = sep_y + 28
info_bottom = info_top + 320
draw.rectangle([(info_x1, info_top), (info_x2, info_bottom)], fill="#ffffff")

# Thin separator above "About this event"
about_sep_y = info_bottom + 20
draw.line([(48, about_sep_y), (1392, about_sep_y)], fill="#ece9ee", width=1)

# About card background area (slightly off-white)
about_top = about_sep_y + 18
about_bottom = about_top + 220
draw.rectangle([(48, about_top), (1392, about_bottom)], fill="#ffffff")

# Large ticket card with blue border (rounded)
ticket_x1, ticket_x2 = 48, 1392
ticket_top = 2320
ticket_bottom = 2510
# outer stroke (blue)
draw.rounded_rectangle(
    [(ticket_x1, ticket_top), (ticket_x2, ticket_bottom)],
    radius=18,
    outline="#2f49ff",
    width=6,
    fill="#ffffff"
)
# inner subtle background to give depth
inner_pad = 12
draw.rounded_rectangle(
    [(ticket_x1 + inner_pad, ticket_top + inner_pad), (ticket_x2 - inner_pad, ticket_bottom - inner_pad)],
    radius=14,
    fill="#ffffff"
)

# Subtle divider above ticket card
draw.line([(48, ticket_top - 28), (1392, ticket_top - 28)], fill="#ece9ee", width=1)

# Note: reserve button area is intentionally left blank (will be pasted on top). 
# Draw a faint bottom area band to match page padding
bottom_band_top = 2660
bottom_band_bottom = 2960
draw.rectangle([(0, bottom_band_top), (1440, bottom_band_bottom)], fill="#ffffff")

# Additional horizontal separators to separate content blocks
draw.line([(48, 920), (1392, 920)], fill="#f3f1f5", width=1)
draw.line([(48, 1620), (1392, 1620)], fill="#f3f1f5", width=1)

# Left and right page margins subtle vertical guides (visual structure only)
draw.line([(40, hdr_bottom + 10), (40, bottom_band_top - 10)], fill="#ffffff", width=1)
draw.line([(1400, hdr_bottom + 10), (1400, bottom_band_top - 10)], fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1195), _c0)
except Exception:
    pass
layout["Following"] = [946, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/01_icon_Food_Drink_._Spirits.png
try:
    _c1 = get_crop(1, 472, 100)
    canvas.paste(_c1, (41, 2071), _c1)
except Exception:
    pass
layout["Food_&_Drink_._Spirits"] = [41, 2071, 513, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/02_icon_Decrease.png
try:
    _c2 = get_crop(2, 99, 96)
    canvas.paste(_c2, (996, 2444), _c2)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/03_icon_9.09.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["9.09"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/04_icon_Reserve_a_spot.png
try:
    _c4 = get_crop(4, 1296, 132)
    canvas.paste(_c4, (72, 2756), _c4)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/05_icon_More.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/06_icon_Increase.png
try:
    _c6 = get_crop(6, 96, 96)
    canvas.paste(_c6, (1224, 2444), _c6)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 91, 102)
    canvas.paste(_c7, (1109, 2442), _c7)
except Exception:
    pass
layout["icon_7"] = [1109, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/08_icon_Share.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1260, 108), _c8)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/09_icon_Ticket_sales_end_soon.png
try:
    _c9 = get_crop(9, 548, 86)
    canvas.paste(_c9, (40, 752), _c9)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 752, 588, 838]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 97, 64)
    canvas.paste(_c10, (1216, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 1, 1313, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 56, 65)
    canvas.paste(_c11, (1317, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1317, 1, 1373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/12_icon_5_00_PM.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1116, 108), _c12)
except Exception:
    pass
layout["5:00_PM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/13_icon_Free.png
try:
    _c13 = get_crop(13, 75, 72)
    canvas.paste(_c13, (249, 2588), _c13)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/14_icon_Free.png
try:
    _c14 = get_crop(14, 138, 106)
    canvas.paste(_c14, (97, 2573), _c14)
except Exception:
    pass
layout["Free"] = [97, 2573, 235, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/15_icon_Unveiling_the_1800_Essential_Artist_Seri.png
try:
    _c15 = get_crop(15, 99, 96)
    canvas.paste(_c15, (996, 2444), _c15)
except Exception:
    pass
layout["Unveiling_the_1800_Essent"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 57)
    canvas.paste(_c16, (316, 6), _c16)
except Exception:
    pass
layout["icon_16"] = [316, 6, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/17_text_9.09.png
try:
    _c17 = get_crop(17, 91, 43)
    canvas.paste(_c17, (20, 17), _c17)
except Exception:
    pass
layout["9.09"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/18_text_Thursday_March_21.png
try:
    _c18 = get_crop(18, 426, 144)
    canvas.paste(_c18, (144, 1155), _c18)
except Exception:
    pass
layout["Thursday;_March_21"] = [144, 1155, 570, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/19_text_5_00_PM.png
try:
    _c19 = get_crop(19, 209, 56)
    canvas.paste(_c19, (567, 893), _c19)
except Exception:
    pass
layout["5:00_PM"] = [567, 893, 776, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/20_text_Tequila_Artistic_Transformation.png
try:
    _c20 = get_crop(20, 426, 144)
    canvas.paste(_c20, (144, 1155), _c20)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [144, 1155, 570, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/21_text_USQ_Wines_Spirits.png
try:
    _c21 = get_crop(21, 426, 144)
    canvas.paste(_c21, (144, 1155), _c21)
except Exception:
    pass
layout["USQ_Wines_&_Spirits"] = [144, 1155, 570, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/22_text_25_Followers.png
try:
    _c22 = get_crop(22, 426, 144)
    canvas.paste(_c22, (144, 1155), _c22)
except Exception:
    pass
layout["25_Followers"] = [144, 1155, 570, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/23_text_Union_Square_Wine_Spirits.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1422), _c23)
except Exception:
    pass
layout["Union_Square_Wine_&_Spiri"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/24_text_2_hrs.png
try:
    _c24 = get_crop(24, 112, 49)
    canvas.paste(_c24, (141, 1580), _c24)
except Exception:
    pass
layout["2_hrs"] = [141, 1580, 253, 1629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/25_text_Refund_policy.png
try:
    _c25 = get_crop(25, 299, 63)
    canvas.paste(_c25, (138, 1685), _c25)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/26_text_The_organizer_will_review_refund_request.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1422), _c26)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_07_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-9/27_text_1800_Tequila_Artistic_Transformation.png
try:
    _c27 = get_crop(27, 75, 72)
    canvas.paste(_c27, (249, 2588), _c27)
except Exception:
    pass
layout["1800_Tequila_&_Artistic_T"] = [249, 2588, 324, 2660]
