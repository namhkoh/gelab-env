# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_07
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9.png
# step_index: 7/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for Eventbrite-like page
width, height = canvas.size

# Colors
bg_color = "#FBFAFC"           # very light off-white background
status_bar_color = "#CFCFCF"   # status bar gray
header_base = "#E6EAEE"        # header base placeholder
header_gradient_dark = (30, 30, 30)  # dark for image overlay gradient (RGB tuple for manual mixing)
card_bg = "#FFFFFF"            # main white card
muted_card = "#F6F6F9"         # light gray card for organizer block
divider = "#EDEAF2"           # subtle divider line
pill_bg = "#F5F3F8"            # subtle pill background (not drawing actual pills)
accent_line = "#5B3A9E"        # not used for text, only subtle accents if needed

# Fill overall background
draw.rectangle((0, 0, width, height), fill=bg_color)

# Status bar (approx 56px height)
status_h = 56
draw.rectangle((0, 0, width, status_h), fill=status_bar_color)

# Header / hero image area (photo will be pasted over this area later)
header_top = status_h
header_bottom = 520
# base fill
draw.rectangle((0, header_top, width, header_bottom), fill=header_base)

# Simulated darkening gradient overlay toward bottom of the header (structural only)
grad_height = 80
for i in range(grad_height):
    # interpolate between transparent-ish and dark overlay by adjusting gray
    t = i / float(grad_height - 1)
    # create a subtle darkening (not too strong so pasted image remains primary)
    gray = int(240 - t * 120)  # from near-white to darker gray
    draw.rectangle((0, header_bottom - grad_height + i, width, header_bottom - grad_height + i + 1),
                   fill=(gray, gray, gray))

# Progress bar track under the header image (thin track across)
pb_y = header_bottom - 36
track_h = 8
track_x0 = 40
track_x1 = width - 40
# draw full track
draw.rounded_rectangle((track_x0, pb_y, track_x1, pb_y + track_h), radius=4, fill="#E6E7E9")
# draw progress segments as subtle darker rectangles (structure only)
seg_w = int((track_x1 - track_x0) * 0.25)
gap = 12
seg_y0 = pb_y + 1
seg_y1 = pb_y + track_h - 1
for i in range(4):
    sx = track_x0 + i * (seg_w + gap)
    draw.rectangle((sx, seg_y0, sx + seg_w - 6, seg_y1), fill="#BFC1C6")

# Main content card (white rounded rectangle) below header
card_top = header_bottom - 20  # slight overlap with image like in real UI
card_left = 24
card_right = width - 24
card_bottom = 2320  # keep above the 'Reserve a spot' auto-pasted area (starts at 2324)
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom), radius=32, fill=card_bg)

# Subtle inner shadow line at top of content card for separation
draw.line((card_left + 8, card_top + 2, card_right - 8, card_top + 2), fill="#F0EEF3", width=2)

# Organizer/follow card background (rounded pill-like rectangle)
org_card_top = 1200
org_card_bottom = org_card_top + 160
org_card_left = 40
org_card_right = card_right - 40
draw.rounded_rectangle((org_card_left, org_card_top, org_card_right, org_card_bottom),
                       radius=28, fill=muted_card)

# Draw a subtle dividing line under the organizer card inside content
divider_y1 = org_card_bottom + 36
draw.line((card_left + 36, divider_y1, card_right - 36, divider_y1), fill=divider, width=2)

# Additional section divider lines to separate event info blocks
seps = [980, 1180, 1700]  # approximate y positions for separators (structural only)
for y in seps:
    # only draw if within the main card region and not overlapping reserved area
    if card_top + 10 < y < card_bottom - 10:
        draw.line((card_left + 36, y, card_right - 36, y), fill=divider, width=1)

# 'About this event' header bar area (structural underline)
about_y = 2080
draw.line((card_left + 36, about_y, card_right - 36, about_y), fill=divider, width=1)

# Small category pill background (structure only, actual text icon will be pasted)
pill_w = 360
pill_h = 56
pill_x = card_left + 36
pill_y = about_y + 28
draw.rounded_rectangle((pill_x, pill_y, pill_x + pill_w, pill_y + pill_h), radius=28, fill=pill_bg)

# Sub-card / ticket selection outline above reserve area (but avoid drawing actual controls)
# Draw only the outline box structure; keep it above the reserve area top (2324)
ticket_box_top = 2220
ticket_box_bottom = 2320 - 8
ticket_box_left = card_left + 36
ticket_box_right = card_right - 36
draw.rounded_rectangle((ticket_box_left, ticket_box_top, ticket_box_right, ticket_box_bottom),
                       radius=12, outline="#3E55D7", width=6, fill=None)

# Final subtle horizontal rule separating top hero and content (just above card_top to create depth)
draw.line((card_left + 8, card_top - 6, card_right - 8, card_top - 6), fill="#E9E7EE", width=2)

# Note: All actual icons, texts, and actionable buttons will be pasted later at their detected positions.
# This drawing only provides background fills, cards, and separators (no duplicated UI content).

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 112, 105)
    canvas.paste(_c1, (988, 2440), _c1)
except Exception:
    pass
layout["icon_1"] = [988, 2440, 1100, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1440, 636)
    canvas.paste(_c3, (0, 2324), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 107, 104)
    canvas.paste(_c4, (1215, 2441), _c4)
except Exception:
    pass
layout["icon_4"] = [1215, 2441, 1322, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/05_icon_Travel_Outdoor_._Hiking.png
try:
    _c5 = get_crop(5, 545, 98)
    canvas.paste(_c5, (40, 2167), _c5)
except Exception:
    pass
layout["Travel_&_Outdoor_._Hiking"] = [40, 2167, 585, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 92, 101)
    canvas.paste(_c6, (1108, 2443), _c6)
except Exception:
    pass
layout["icon_6"] = [1108, 2443, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/07_icon_Share.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 108), _c7)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/08_icon_Few_tickets_left.png
try:
    _c8 = get_crop(8, 428, 84)
    canvas.paste(_c8, (41, 754), _c8)
except Exception:
    pass
layout["Few_tickets_left"] = [41, 754, 469, 838]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/09_icon_4.36.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (36, 108), _c9)
except Exception:
    pass
layout["4.36"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 56, 64)
    canvas.paste(_c10, (1317, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1317, 1, 1373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/11_icon_Sports_Basement.png
try:
    _c11 = get_crop(11, 370, 144)
    canvas.paste(_c11, (288, 1250), _c11)
except Exception:
    pass
layout["Sports_Basement"] = [288, 1250, 658, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/12_icon_4.36.png
try:
    _c12 = get_crop(12, 64, 70)
    canvas.paste(_c12, (179, 1), _c12)
except Exception:
    pass
layout["4.36"] = [179, 1, 243, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/13_icon_Ticket_sales_end_soon.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1116, 108), _c13)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/14_icon_4.36.png
try:
    _c14 = get_crop(14, 65, 70)
    canvas.paste(_c14, (113, 0), _c14)
except Exception:
    pass
layout["4.36"] = [113, 0, 178, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/15_icon_Ticket_sales_end_soon.png
try:
    _c15 = get_crop(15, 550, 84)
    canvas.paste(_c15, (472, 753), _c15)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [472, 753, 1022, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 68, 70)
    canvas.paste(_c16, (307, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [307, 0, 375, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 57, 69)
    canvas.paste(_c17, (246, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [246, 1, 303, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 81, 62)
    canvas.paste(_c18, (1215, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [1215, 2, 1296, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 44, 60)
    canvas.paste(_c19, (1271, 4), _c19)
except Exception:
    pass
layout["icon_19"] = [1271, 4, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 51, 71)
    canvas.paste(_c20, (382, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 0, 433, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/21_icon_Free.png
try:
    _c21 = get_crop(21, 135, 106)
    canvas.paste(_c21, (100, 2575), _c21)
except Exception:
    pass
layout["Free"] = [100, 2575, 235, 2681]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/22_text_Saturday_April_27.png
try:
    _c22 = get_crop(22, 449, 77)
    canvas.paste(_c22, (38, 885), _c22)
except Exception:
    pass
layout["Saturday,_April_27"] = [38, 885, 487, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/23_text_1I_00AM.png
try:
    _c23 = get_crop(23, 241, 56)
    canvas.paste(_c23, (523, 893), _c23)
except Exception:
    pass
layout["1I:00AM"] = [523, 893, 764, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/24_text_Backpacking_Clinic_w_Sports_Basement.png
try:
    _c24 = get_crop(24, 370, 144)
    canvas.paste(_c24, (288, 1250), _c24)
except Exception:
    pass
layout["Backpacking_Clinic_w__Spo"] = [288, 1250, 658, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/25_text_Sports_Basement_Berkeley.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 1517), _c25)
except Exception:
    pass
layout["Sports_Basement_Berkeley"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/26_text_4hrs.png
try:
    _c26 = get_crop(26, 112, 50)
    canvas.paste(_c26, (141, 1674), _c26)
except Exception:
    pass
layout["4hrs"] = [141, 1674, 253, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/27_text_Refund_policy.png
try:
    _c27 = get_crop(27, 299, 63)
    canvas.paste(_c27, (138, 1780), _c27)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/28_text_The_organizer_will_review_refund_request.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 1517), _c28)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/29_text_About_this_event.png
try:
    _c29 = get_crop(29, 452, 57)
    canvas.paste(_c29, (46, 2081), _c29)
except Exception:
    pass
layout["About_this_event"] = [46, 2081, 498, 2138]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/30_text_General_Admission.png
try:
    _c30 = get_crop(30, 415, 55)
    canvas.paste(_c30, (116, 2451), _c30)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_07_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-9/31_clickable_Organizer_profile_picture.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (96, 1289), _c31)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1289, 240, 1433]
