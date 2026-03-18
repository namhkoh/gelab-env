# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_21
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-23.png
# step_index: 21/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")

# Status bar area (top ~56px)
draw.rectangle([(0, 0), (1440, 56)], fill="#CFD0D2")

# Top banner / hero image background (dark blurred area)
banner_top = 56
banner_bottom = 430
draw.rectangle([(0, banner_top), (1440, banner_bottom)], fill="#2B2640")
# Vignette sides to mimic blurred photo edges
draw.rectangle([(0, banner_top), (220, banner_bottom)], fill="#1F1726")
draw.rectangle([(1220, banner_top), (1440, banner_bottom)], fill="#1F1726")

# Soft overlay band near bottom of banner to hint transition to content
draw.rectangle([(0, banner_bottom - 48), (1440, banner_bottom)], fill="#2F2940")

# Thin divider under banner
draw.line([(48, banner_bottom + 14), (1392, banner_bottom + 14)], fill="#EDEBF1", width=1)

# Organizer / profile card background (rounded)
org_card_x1, org_card_y1 = 48, 980
org_card_x2, org_card_y2 = 1392, 1128
# subtle shadow
draw.rounded_rectangle([(org_card_x1 + 6, org_card_y1 + 8), (org_card_x2 + 6, org_card_y2 + 8)],
                       radius=24, fill="#F0EFF3")
# card
draw.rounded_rectangle([(org_card_x1, org_card_y1), (org_card_x2, org_card_y2)],
                       radius=24, fill="#F6F6F8", outline="#E6E4EA", width=2)

# Details section background (white area is main canvas; add a very subtle band)
details_band_y = org_card_y2 + 40
draw.rectangle([(0, details_band_y), (1440, details_band_y + 420)], fill="#FFFFFF")

# Light divider line between details and date selector
divider_y = details_band_y + 360
draw.line([(48, divider_y), (1392, divider_y)], fill="#E9E7EB", width=2)

# "Select date and time" cards row (rounded cards as placeholders)
date_row_y1 = divider_y + 36
date_row_height = 256
card_w = 330
gap = 36
start_x = 48
for i in range(4):
    x1 = start_x + i * (card_w + gap)
    x2 = x1 + card_w
    # light card background
    draw.rounded_rectangle([(x1, date_row_y1), (x2, date_row_y1 + date_row_height)],
                           radius=18, fill="#FFFFFF", outline="#ECEAF4", width=2)
    # subtle top underline to indicate label area
    draw.line([(x1 + 24, date_row_y1 + 64), (x2 - 24, date_row_y1 + 64)], fill="#F2F1F6", width=2)

# Thin separator above ticket area
ticket_top = date_row_y1 + date_row_height + 32
draw.line([(48, ticket_top), (1392, ticket_top)], fill="#E9E7EB", width=1)

# Ticket selection card (rounded with subtle highlight border)
ticket_card_y1 = ticket_top + 24
ticket_card_y2 = ticket_card_y1 + 120
draw.rounded_rectangle([(48, ticket_card_y1 + 6), (1392, ticket_card_y2 + 6)],
                       radius=18, fill="#F7F7FA")
draw.rounded_rectangle([(48, ticket_card_y1), (1392, ticket_card_y2)],
                       radius=18, fill="#FFFFFF", outline="#3B55FF", width=4)

# Very subtle horizontal separators inside content area to hint sections
for y in (ticket_card_y2 + 32, ticket_card_y2 + 96, ticket_card_y2 + 180):
    draw.line([(48, y), (1392, y)], fill="#F0EFF3", width=1)

# Keep bottom area clear for the "Reserve a spot" overlay (do NOT draw there).
# Provide a subtle top handle for the reserve sheet (rounded white notch) just above it
reserve_sheet_top = 2324
handle_w, handle_h = 120, 8
handle_x = (1440 - handle_w) // 2
draw.rounded_rectangle([(handle_x, reserve_sheet_top - 18), (handle_x + handle_w, reserve_sheet_top - 10)],
                       radius=8, fill="#EFEFF1")

# Done - all drawn elements are background/structure only.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1163), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/01_icon_May.png
try:
    _c1 = get_crop(1, 450, 257)
    canvas.paste(_c1, (474, 2067), _c1)
except Exception:
    pass
layout["May"] = [474, 2067, 924, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/02_icon_April.png
try:
    _c2 = get_crop(2, 450, 257)
    canvas.paste(_c2, (24, 2067), _c2)
except Exception:
    pass
layout["April"] = [24, 2067, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/03_icon_May.png
try:
    _c3 = get_crop(3, 450, 257)
    canvas.paste(_c3, (924, 2067), _c3)
except Exception:
    pass
layout["May"] = [924, 2067, 1374, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/04_icon_More.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1116, 108), _c4)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/05_icon_May.png
try:
    _c5 = get_crop(5, 110, 104)
    canvas.paste(_c5, (988, 2440), _c5)
except Exception:
    pass
layout["May"] = [988, 2440, 1098, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/06_icon_Share.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1260, 108), _c6)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 107, 103)
    canvas.paste(_c7, (1215, 2442), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 2442, 1322, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/08_icon_Reserve_a_spot.png
try:
    _c8 = get_crop(8, 1440, 636)
    canvas.paste(_c8, (0, 2324), _c8)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/09_icon_7.30.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (36, 108), _c9)
except Exception:
    pass
layout["7.30"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/10_icon_May.png
try:
    _c10 = get_crop(10, 90, 101)
    canvas.paste(_c10, (1109, 2443), _c10)
except Exception:
    pass
layout["May"] = [1109, 2443, 1199, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 46, 67)
    canvas.paste(_c11, (1156, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1156, 2, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/12_icon_7.30.png
try:
    _c12 = get_crop(12, 63, 68)
    canvas.paste(_c12, (180, 1), _c12)
except Exception:
    pass
layout["7.30"] = [180, 1, 243, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 67)
    canvas.paste(_c13, (246, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 1, 302, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 46, 63)
    canvas.paste(_c14, (1327, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [1327, 3, 1373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/15_icon_Free.png
try:
    _c15 = get_crop(15, 134, 103)
    canvas.paste(_c15, (100, 2576), _c15)
except Exception:
    pass
layout["Free"] = [100, 2576, 234, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 67, 67)
    canvas.paste(_c16, (308, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [308, 1, 375, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/17_icon_7.30.png
try:
    _c17 = get_crop(17, 61, 68)
    canvas.paste(_c17, (115, 0), _c17)
except Exception:
    pass
layout["7.30"] = [115, 0, 176, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 51, 67)
    canvas.paste(_c18, (382, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 1, 433, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/19_icon_Free.png
try:
    _c19 = get_crop(19, 102, 110)
    canvas.paste(_c19, (233, 2575), _c19)
except Exception:
    pass
layout["Free"] = [233, 2575, 335, 2685]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 100, 64)
    canvas.paste(_c20, (1215, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [1215, 2, 1315, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/21_icon_6.30_PM.png
try:
    _c21 = get_crop(21, 445, 144)
    canvas.paste(_c21, (144, 1123), _c21)
except Exception:
    pass
layout["6.30_PM"] = [144, 1123, 589, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/22_text_7.30.png
try:
    _c22 = get_crop(22, 89, 43)
    canvas.paste(_c22, (22, 17), _c22)
except Exception:
    pass
layout["7.30"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/23_text_Marcelo_Maccagnan.png
try:
    _c23 = get_crop(23, 445, 144)
    canvas.paste(_c23, (144, 1123), _c23)
except Exception:
    pass
layout["Marcelo_Maccagnan"] = [144, 1123, 589, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/24_text_23_Followers.png
try:
    _c24 = get_crop(24, 445, 144)
    canvas.paste(_c24, (144, 1123), _c24)
except Exception:
    pass
layout["23_Followers"] = [144, 1123, 589, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/25_text_pinkFROG_cafe.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 1390), _c25)
except Exception:
    pass
layout["pinkFROG_cafe"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/26_text_3_hrs.png
try:
    _c26 = get_crop(26, 114, 50)
    canvas.paste(_c26, (139, 1547), _c26)
except Exception:
    pass
layout["3_hrs"] = [139, 1547, 253, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/27_text_Refund_policy.png
try:
    _c27 = get_crop(27, 299, 63)
    canvas.paste(_c27, (138, 1653), _c27)
except Exception:
    pass
layout["Refund_policy"] = [138, 1653, 437, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/28_text_The_organizer_will_review_refund_request.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 1390), _c28)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/29_text_Select_date_and_time.png
try:
    _c29 = get_crop(29, 450, 257)
    canvas.paste(_c29, (24, 2067), _c29)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 2067, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_21_2024_4_23_19_27_45f56b06f31541079045047b6d542613-23/30_text_General_Admission.png
try:
    _c30 = get_crop(30, 450, 257)
    canvas.paste(_c30, (24, 2067), _c30)
except Exception:
    pass
layout["General_Admission"] = [24, 2067, 474, 2324]
