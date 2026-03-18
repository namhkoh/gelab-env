# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_13
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15.png
# step_index: 13/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for 1440x2960 canvas
# Uses available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill overall background with a warm cream color
draw.rectangle([(0, 0), (1440, 2960)], fill=(249, 244, 238))

# Status bar area at the very top (~84px high) - subtle muted gray
status_h = 84
draw.rectangle([(0, 0), (1440, status_h)], fill=(221, 221, 221))

# Header/banner area (decorative image background)
banner_y0 = status_h
banner_y1 = 520
banner_pad_x = 40
banner_outer = (banner_pad_x, banner_y0 + 20, 1440 - banner_pad_x, banner_y1)
# Outer rounded frame with soft gold outline
draw.rounded_rectangle(banner_outer, radius=18, fill=(252, 249, 246), outline=(204, 167, 102), width=6)
# Inner subtle lighter area to mimic image canvas
inner_inset = 28
inner_rect = (banner_outer[0] + inner_inset, banner_outer[1] + inner_inset,
              banner_outer[2] - inner_inset, banner_outer[3] - inner_inset)
draw.rounded_rectangle(inner_rect, radius=12, fill=(255, 255, 255))

# Main white content area below the banner (rounded top corners)
content_y0 = banner_y1 + 20
content_y1 = 2200  # stop above ticket purchase area (tickets area will be pasted later)
draw.rounded_rectangle((0, content_y0, 1440, content_y1), radius=28, fill=(255, 255, 255))

# Light drop shadow for content area (very subtle)
shadow_top = content_y0
draw.rectangle([(0, shadow_top - 4), (1440, shadow_top)], fill=(230, 230, 230))

# Organizer / follow card background (rounded rectangle)
# Place it near the top of the content area (matches where the Follow button will be pasted)
org_card_x0 = 40
org_card_x1 = 1400
org_card_h = 140
org_card_y0 = 1240
org_card_y1 = org_card_y0 + org_card_h
# shadow under card
draw.rectangle([(org_card_x0 + 6, org_card_y0 + 8), (org_card_x1 + 6, org_card_y1 + 8)], fill=(236, 233, 236))
# card fill (very light gray/purple tint)
draw.rounded_rectangle((org_card_x0, org_card_y0, org_card_x1, org_card_y1), radius=28, fill=(249, 247, 250))

# Thin divider line under organizer card
draw.line([(org_card_x0 + 8, org_card_y1 + 18), (org_card_x1 - 8, org_card_y1 + 18)], fill=(236, 234, 238), width=1)

# Small informational pill (e.g., "ticket sales end soon" badge background)
pill_x0 = 40
pill_y0 = 700
pill_x1 = 360
pill_y1 = pill_y0 + 56
draw.rounded_rectangle((pill_x0, pill_y0, pill_x1, pill_y1), radius=28, fill=(243, 237, 255))
# subtle pill border
draw.rounded_rectangle((pill_x0, pill_y0, pill_x1, pill_y1), radius=28, outline=(217, 198, 255), width=2)

# Separator line between info section and "About this event"
sep_y = 1860
draw.line([(40, sep_y), (1400, sep_y)], fill=(238, 236, 239), width=2)

# Small subtle accent under the section header area (to anchor the "About this event")
about_accent_y = 2020
draw.line([(40, about_accent_y), (220, about_accent_y)], fill=(225, 222, 230), width=6)

# Decorative rounded tag background for category tags (behind auto-pasted text tags)
tag_x0 = 48
tag_y0 = 2040
tag_x1 = 460
tag_y1 = tag_y0 + 56
draw.rounded_rectangle((tag_x0, tag_y0, tag_x1, tag_y1), radius=28, fill=(246, 245, 249))

# Ensure we do not draw anything in the ticket/purchase region (y >= 2276) per instructions.
# The reserved spot / ticket purchase area will be pasted later by the pipeline.

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 112, 105)
    canvas.paste(_c1, (987, 2393), _c1)
except Exception:
    pass
layout["icon_1"] = [987, 2393, 1099, 2498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1440, 684)
    canvas.paste(_c2, (0, 2276), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2276, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 109, 104)
    canvas.paste(_c3, (1215, 2394), _c3)
except Exception:
    pass
layout["icon_3"] = [1215, 2394, 1324, 2498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 94, 104)
    canvas.paste(_c4, (1107, 2393), _c4)
except Exception:
    pass
layout["icon_4"] = [1107, 2393, 1201, 2497]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/05_icon_More.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/06_icon_fev.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1116, 108), _c6)
except Exception:
    pass
layout["fev"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/07_icon_5.16.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (36, 108), _c7)
except Exception:
    pass
layout["5.16"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/08_icon_Free.png
try:
    _c8 = get_crop(8, 134, 94)
    canvas.paste(_c8, (100, 2578), _c8)
except Exception:
    pass
layout["Free"] = [100, 2578, 234, 2672]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/09_icon_Share.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1260, 108), _c9)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/10_icon_Free.png
try:
    _c10 = get_crop(10, 97, 104)
    canvas.paste(_c10, (237, 2574), _c10)
except Exception:
    pass
layout["Free"] = [237, 2574, 334, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/11_icon_Ticket_sales_end_soon.png
try:
    _c11 = get_crop(11, 547, 84)
    canvas.paste(_c11, (40, 753), _c11)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/12_icon_5.16.png
try:
    _c12 = get_crop(12, 64, 69)
    canvas.paste(_c12, (179, 1), _c12)
except Exception:
    pass
layout["5.16"] = [179, 1, 243, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 99, 59)
    canvas.paste(_c13, (1215, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [1215, 4, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/14_icon_Charity_Causes.png
try:
    _c14 = get_crop(14, 1440, 684)
    canvas.paste(_c14, (0, 2276), _c14)
except Exception:
    pass
layout["Charity_&_Causes"] = [0, 2276, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 57, 66)
    canvas.paste(_c15, (246, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [246, 1, 303, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 43, 59)
    canvas.paste(_c16, (1328, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [1328, 3, 1371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/17_icon_SUNDAY_APRIL_28T0_11AM-3PM.png
try:
    _c17 = get_crop(17, 72, 65)
    canvas.paste(_c17, (305, 1), _c17)
except Exception:
    pass
layout["SUNDAY;_APRIL_28T0_|_11AM"] = [305, 1, 377, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/18_icon_Animal_Care_League_s_27th_Annual.png
try:
    _c18 = get_crop(18, 433, 144)
    canvas.paste(_c18, (144, 1290), _c18)
except Exception:
    pass
layout["Animal_Care_League's_27th"] = [144, 1290, 577, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/19_icon_Spring_for_the_Animals_Luncheon.png
try:
    _c19 = get_crop(19, 433, 144)
    canvas.paste(_c19, (144, 1290), _c19)
except Exception:
    pass
layout["Spring_for_the_Animals_Lu"] = [144, 1290, 577, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/20_text_5.16.png
try:
    _c20 = get_crop(20, 89, 43)
    canvas.paste(_c20, (22, 17), _c20)
except Exception:
    pass
layout["5.16"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/21_text_The_Oak_Park_Country_Club.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1517), _c21)
except Exception:
    pass
layout["The_Oak_Park_Country_Club"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/22_text_4hrs.png
try:
    _c22 = get_crop(22, 112, 50)
    canvas.paste(_c22, (141, 1674), _c22)
except Exception:
    pass
layout["4hrs"] = [141, 1674, 253, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/23_text_Refund_policy.png
try:
    _c23 = get_crop(23, 299, 63)
    canvas.paste(_c23, (138, 1780), _c23)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/24_text_The_organizer_will_review_refund_request.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1517), _c24)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/25_text_About_this_event.png
try:
    _c25 = get_crop(25, 454, 61)
    canvas.paste(_c25, (45, 2080), _c25)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 499, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/26_text_Buy_tickets_at_the_link_below_Single.png
try:
    _c26 = get_crop(26, 1440, 684)
    canvas.paste(_c26, (0, 2276), _c26)
except Exception:
    pass
layout["Buy_tickets_at_the_link_b"] = [0, 2276, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_13_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-15/27_text_ticket_S80.png
try:
    _c27 = get_crop(27, 244, 61)
    canvas.paste(_c27, (110, 2471), _c27)
except Exception:
    pass
layout["ticket:_S80"] = [110, 2471, 354, 2532]
