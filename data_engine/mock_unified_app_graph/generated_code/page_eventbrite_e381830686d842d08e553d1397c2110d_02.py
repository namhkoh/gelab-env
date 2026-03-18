# page_id: page_eventbrite_e381830686d842d08e553d1397c2110d_02
# screenshot: 2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4.png
# step_index: 2/3
# task: Open Eventbrite. Open "Recommended". Select the third recommended event. Add it to favourites. What is the refund policy?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background, status bar, headers, section cards, separators, and ticket area backgrounds
# Uses provided canvas (PIL Image) and draw (ImageDraw)

# Full background fill (slightly off-white to match UI)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")

# Status bar area (top ~64px)
status_h = 64
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")
# subtle top notch line and bottom divider for status bar
draw.line([(0, 0), (1440, 0)], fill="#BDBDBD", width=1)
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill="#BFBFC1", width=1)

# Header / cover image background area (beneath status bar)
header_top = status_h
header_bottom = 460
header_h = header_bottom - header_top

# horizontal gradient for header (dark teal to deep slate)
start_rgb = (41, 70, 72)   # dark teal
end_rgb = (38, 102, 102)   # slightly greener teal
for i in range(header_h):
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * (i / max(1, header_h - 1)))
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * (i / max(1, header_h - 1)))
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * (i / max(1, header_h - 1)))
    draw.line([(0, header_top + i), (1440, header_top + i)], fill=(r, g, b))

# Decorative slanted strip at bottom of header (teal band)
strip_color = (64, 112, 122)
draw.polygon([
    (0, header_bottom - 56),
    (1440, header_bottom - 112),
    (1440, header_bottom),
    (0, header_bottom)
], fill=strip_color)

# Subtle divider below header
draw.line([(32, header_bottom + 8), (1440 - 32, header_bottom + 8)], fill="#ECEAF1", width=2)

# Main content area remains white; draw faint top margin shadow under header
draw.rectangle([(0, header_bottom + 8), (1440, header_bottom + 14)], fill="#F3F3F6")

# Organizer / host card (rounded rectangle background)
org_x1, org_y1 = 48, 720
org_x2, org_y2 = 1392, 864
org_radius = 28
# subtle shadow (a slightly darker rounded rect offset)
shadow_offset = 6
draw.rounded_rectangle(
    [(org_x1, org_y1 + shadow_offset), (org_x2, org_y2 + shadow_offset)],
    radius=org_radius, fill="#F6F6F8"
)
# main card
draw.rounded_rectangle(
    [(org_x1, org_y1), (org_x2, org_y2)],
    radius=org_radius, fill="#FFFFFF", outline="#ECEAF1", width=2
)

# Divider lines and small separators in the content
# 1) under the title/content area
divider_y1 = 1160
draw.line([(32, divider_y1), (1440 - 32, divider_y1)], fill="#EFEFF2", width=1)

# 2) light thin divider between info rows
row_start_x = 48
row_end_x = 1392
row_y = 1360
draw.line([(row_start_x, row_y), (row_end_x, row_y)], fill="#F0EDF3", width=1)

# "About this event" separator (faint)
about_sep_y = 1540
draw.line([(32, about_sep_y), (1440 - 32, about_sep_y)], fill="#F1F1F4", width=2)

# Ticket selection card background (rounded, with blue outline) - positioned above Reserve area
# Reserve area begins around y=2324 (detected). Draw ticket card inside that region as background.
ticket_x1, ticket_y1 = 40, 2380
ticket_x2, ticket_y2 = 1400, 2536
ticket_radius = 20
# shadow
draw.rounded_rectangle(
    [(ticket_x1, ticket_y1 + 6), (ticket_x2, ticket_y2 + 6)],
    radius=ticket_radius, fill="#F5F5F7"
)
# main ticket card with blue border
draw.rounded_rectangle(
    [(ticket_x1, ticket_y1), (ticket_x2, ticket_y2)],
    radius=ticket_radius, fill="#FFFFFF", outline="#3B5AEA", width=6
)

# Thin internal horizontal guide inside ticket area (to suggest separation)
draw.line([(ticket_x1 + 28, (ticket_y1 + ticket_y2) // 2),
           (ticket_x2 - 28, (ticket_y1 + ticket_y2) // 2)], fill="#F1F1F6", width=1)

# Large reserve area background band at bottom (kept neutral so pasted orange button stands out)
reserve_band_y = 2324
draw.rectangle([(0, reserve_band_y), (1440, 2960)], fill="#FFFFFF")
# subtle top divider for reserve band
draw.line([(24, reserve_band_y), (1440 - 24, reserve_band_y)], fill="#EDEEF2", width=2)

# Page-wide subtle vertical padding lines / rules for layout balance
# left margin guide (faint) and right margin
draw.line([(48, 0), (48, 2960)], fill="#FBFBFD", width=1)
draw.line([(1392, 0), (1392, 2960)], fill="#FBFBFD", width=1)

# Final subtle overall vignette at very bottom to ground the reserve area
draw.rectangle([(0, 2860), (1440, 2960)], fill="#FCFBFD")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/01_icon_Understanding.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["Understanding"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 112, 105)
    canvas.paste(_c2, (988, 2440), _c2)
except Exception:
    pass
layout["icon_2"] = [988, 2440, 1100, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 105, 103)
    canvas.paste(_c3, (1217, 2441), _c3)
except Exception:
    pass
layout["icon_3"] = [1217, 2441, 1322, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/04_icon_7.02.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["7.02"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 93, 103)
    canvas.paste(_c5, (1108, 2441), _c5)
except Exception:
    pass
layout["icon_5"] = [1108, 2441, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/06_icon_Reserve_a_spot.png
try:
    _c6 = get_crop(6, 1440, 636)
    canvas.paste(_c6, (0, 2324), _c6)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/07_icon_Share.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 108), _c7)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/08_icon_7.02.png
try:
    _c8 = get_crop(8, 64, 63)
    canvas.paste(_c8, (179, 2), _c8)
except Exception:
    pass
layout["7.02"] = [179, 2, 243, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/09_icon_7.02.png
try:
    _c9 = get_crop(9, 59, 64)
    canvas.paste(_c9, (116, 1), _c9)
except Exception:
    pass
layout["7.02"] = [116, 1, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/10_icon_grief.png
try:
    _c10 = get_crop(10, 234, 144)
    canvas.paste(_c10, (48, 2090), _c10)
except Exception:
    pass
layout["grief"] = [48, 2090, 282, 2234]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 43, 59)
    canvas.paste(_c11, (1327, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [1327, 4, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 43, 56)
    canvas.paste(_c12, (1271, 6), _c12)
except Exception:
    pass
layout["icon_12"] = [1271, 6, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 65, 58)
    canvas.paste(_c13, (1217, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [1217, 4, 1282, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 60)
    canvas.paste(_c14, (247, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 3, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 61, 61)
    canvas.paste(_c15, (311, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [311, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 64)
    canvas.paste(_c16, (382, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 2, 432, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/17_icon_Free.png
try:
    _c17 = get_crop(17, 135, 100)
    canvas.paste(_c17, (100, 2578), _c17)
except Exception:
    pass
layout["Free"] = [100, 2578, 235, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/18_icon_Free.png
try:
    _c18 = get_crop(18, 103, 113)
    canvas.paste(_c18, (233, 2573), _c18)
except Exception:
    pass
layout["Free"] = [233, 2573, 336, 2686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/19_text_7.02.png
try:
    _c19 = get_crop(19, 89, 43)
    canvas.paste(_c19, (22, 17), _c19)
except Exception:
    pass
layout["7.02"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/20_text_Grief_and_Loss.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1116, 108), _c20)
except Exception:
    pass
layout["Grief_and_Loss"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/21_text_Institute.png
try:
    _c21 = get_crop(21, 125, 30)
    canvas.paste(_c21, (233, 566), _c21)
except Exception:
    pass
layout["Institute"] = [233, 566, 358, 596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/22_text_Wednesday_June_26.png
try:
    _c22 = get_crop(22, 379, 144)
    canvas.paste(_c22, (288, 1028), _c22)
except Exception:
    pass
layout["Wednesday,_June_26"] = [288, 1028, 667, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/23_text_1O_00_AM.png
try:
    _c23 = get_crop(23, 239, 54)
    canvas.paste(_c23, (585, 766), _c23)
except Exception:
    pass
layout["1O:00_AM"] = [585, 766, 824, 820]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/24_text_Understanding_Grief_and_Loss.png
try:
    _c24 = get_crop(24, 379, 144)
    canvas.paste(_c24, (288, 1028), _c24)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [288, 1028, 667, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/25_text_Instit.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (96, 1067), _c25)
except Exception:
    pass
layout["Instit"] = [96, 1067, 240, 1211]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/26_text_Institute_on.png
try:
    _c26 = get_crop(26, 379, 144)
    canvas.paste(_c26, (288, 1028), _c26)
except Exception:
    pass
layout["Institute_on"] = [288, 1028, 667, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/27_text_1.3k_Followers.png
try:
    _c27 = get_crop(27, 379, 144)
    canvas.paste(_c27, (288, 1028), _c27)
except Exception:
    pass
layout["1.3k_Followers"] = [288, 1028, 667, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/28_text_Online_event.png
try:
    _c28 = get_crop(28, 274, 55)
    canvas.paste(_c28, (139, 1341), _c28)
except Exception:
    pass
layout["Online_event"] = [139, 1341, 413, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/29_text_hrs.png
try:
    _c29 = get_crop(29, 77, 50)
    canvas.paste(_c29, (176, 1452), _c29)
except Exception:
    pass
layout["hrs"] = [176, 1452, 253, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/30_text_Refund_policy.png
try:
    _c30 = get_crop(30, 299, 63)
    canvas.paste(_c30, (138, 1558), _c30)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/31_text_The_organizer_will_review_refund_request.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 1295), _c31)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_02_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-4/32_text_General_Admission.png
try:
    _c32 = get_crop(32, 415, 55)
    canvas.paste(_c32, (116, 2451), _c32)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]
