# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_10
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12.png
# step_index: 10/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background (slightly warm white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")

# STATUS BAR (approx height 56) - subtle grey bar across top
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFD1")
# divider below status bar
draw.line([(0, status_h), (1440, status_h)], fill="#BFBFC1", width=1)

# HEADER / HERO IMAGE AREA (banner) - gradient sky to deep blue then dark bottom band
hero_top = status_h
hero_bottom = 420
for i in range(hero_top, hero_bottom):
    # compute t from 0..1
    t = (i - hero_top) / max(1, (hero_bottom - hero_top))
    # gradient from warm pinkish (top) to sky blue to deep navy at bottom
    if t < 0.5:
        # top half: pink -> light blue
        tt = t / 0.5
        r = int(255 * (1 - 0.35 * tt))        # 255 -> ~166
        g = int(150 + (200 - 150) * tt)      # 150 -> 200
        b = int(180 + (230 - 180) * tt)      # 180 -> 230
    else:
        # bottom half: light blue -> dark navy
        tt = (t - 0.5) / 0.5
        r = int(166 - (166 - 20) * tt)       # -> 20
        g = int(200 - (200 - 32) * tt)       # -> 32
        b = int(230 - (230 - 50) * tt)       # -> 50
    draw.line([(0, i), (1440, i)], fill=(r, g, b))

# soft dark overlay at bottom of hero to emulate image crop fade
overlay_height = 80
for j in range(overlay_height):
    alpha = int(200 * (j / overlay_height))  # 0..200
    y = hero_bottom - overlay_height + j
    # composite dark line - approximate by darkening existing pixel row with rectangle line
    draw.line([(0, y), (1440, y)], fill=(int(10*(alpha/200)), int(12*(alpha/200)), int(15*(alpha/200))))

# thin divider under hero
draw.line([(24, hero_bottom + 6), (1440 - 24, hero_bottom + 6)], fill="#E9E6EA", width=1)

# MAIN CONTENT AREA background - keep white but slightly off-white band for grouping
content_x = 24
content_w = 1440 - content_x * 2

# Organizer card (rounded) - light tinted card behind organizer row
org_card_y = 1088
org_card_h = 142
org_card_bbox = (content_x, org_card_y, content_x + content_w, org_card_y + org_card_h)
draw.rounded_rectangle(org_card_bbox, radius=22, fill="#F7F5F8", outline="#ECE7EE", width=1)

# slight shadow under organizer card
shadow_top = org_card_y + org_card_h
for s in range(6):
    alpha = int(50 * (1 - s / 6))
    y = shadow_top + s
    draw.line([(content_x + 6, y), (content_x + content_w - 6, y)], fill=(220, 216, 226))

# Section separators (thin subtle rules) - between info sections
sep1_y = 1680
sep2_y = 2360
draw.line([(24, sep1_y), (1440 - 24, sep1_y)], fill="#ECE8EC", width=2)
draw.line([(24, sep2_y), (1440 - 24, sep2_y)], fill="#ECE8EC", width=2)

# Light section card behind "About this event" content area (rounded pale)
about_y = 1920
about_h = 300
about_bbox = (24, about_y, 1440 - 24, about_y + about_h)
draw.rounded_rectangle(about_bbox, radius=18, fill="#FFFFFF", outline=None)

# subtle pill background hint (do not draw text) - a rounded pale pill where tags appear
pill_x = 48
pill_y = 2064
pill_w = 420
pill_h = 56
draw.rounded_rectangle((pill_x, pill_y, pill_x + pill_w, pill_y + pill_h), radius=28, fill="#F0EFF4")

# Location section background - keep white but add a faint divider line above and below to structure
loc_top = 2520
draw.rectangle([(24, loc_top - 10), (1440 - 24, loc_top + 180)], fill="#FFFFFF")
draw.line([(24, loc_top - 10), (1440 - 24, loc_top - 10)], fill="#EDE9EE", width=1)
draw.line([(24, loc_top + 170), (1440 - 24, loc_top + 170)], fill="#EDE9EE", width=1)

# Bottom fixed ticket bar background (light) - keep area reserved but subtle
bottom_bar_y = 2680
draw.rectangle([(0, bottom_bar_y), (1440, 2960)], fill="#FBF7F8")
# top divider for bottom bar
draw.line([(0, bottom_bar_y), (1440, bottom_bar_y)], fill="#E7E2E6", width=1)

# Left area highlight in bottom bar (price group) - subtle white card on left
left_card_bbox = (36, bottom_bar_y + 20, 460, bottom_bar_y + 116)
draw.rounded_rectangle(left_card_bbox, radius=14, fill="#FFFFFF", outline="#E6E0E5", width=1)

# Right area reserved (do not draw button contents) - draw a faint drop shadow to indicate elevated button region
right_btn_region = (822, bottom_bar_y + 20, 1410, bottom_bar_y + 116)
# subtle raised capsule shadow
for s in range(4):
    alpha = int(40 * (1 - s / 4))
    y = bottom_bar_y + 20 + s
    draw.line([(822 + 4, y), (1410 - 4, y)], fill=(210, 85, 30) if s == 3 else (230, 230, 230))

# Small visual flourishes: left fab/back button faint circular background near top-left of hero
# Keep subtle so icons pasted above remain clear
back_circle_bbox = (24, status_h + 14, 24 + 68, status_h + 14 + 68)
draw.ellipse(back_circle_bbox, fill="#FFFFFF", outline="#F0EDF1", width=1)
# right top small circular backgrounds for heart/share icons (no icons drawn)
right_circle1 = (1320 - 80, status_h + 14, 1320 - 12, status_h + 14 + 68)
right_circle2 = (1240 - 80, status_h + 14, 1240 - 12, status_h + 14 + 68)
draw.ellipse(right_circle1, fill="#FFFFFF", outline="#F0EDF1", width=1)
draw.ellipse(right_circle2, fill="#FFFFFF", outline="#F0EDF1", width=1)

# subtle overall vertical rhythm lines (very light) to guide layout (do not interfere with detected content)
for y in (hero_bottom + 12, org_card_y + org_card_h + 30, sep1_y + 12, sep2_y + 12):
    draw.line([(36, y), (1440 - 36, y)], fill="#F6F4F7", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1195), _c0)
except Exception:
    pass
layout["Following"] = [946, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/02_icon_4.33_my.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["4.33_my"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/04_icon_Celebrate_Art_at_San_Francisco_s_Premier.png
try:
    _c4 = get_crop(4, 234, 144)
    canvas.paste(_c4, (48, 2332), _c4)
except Exception:
    pass
layout["Celebrate_Art_at_San_Fran"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/05_icon_Performing_Visual_Arts.png
try:
    _c5 = get_crop(5, 234, 144)
    canvas.paste(_c5, (48, 2332), _c5)
except Exception:
    pass
layout["Performing_&_Visual_Arts"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/06_icon_4.33_my.png
try:
    _c6 = get_crop(6, 65, 70)
    canvas.paste(_c6, (178, 0), _c6)
except Exception:
    pass
layout["4.33_my"] = [178, 0, 243, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/07_icon_4.33_my.png
try:
    _c7 = get_crop(7, 65, 70)
    canvas.paste(_c7, (112, 0), _c7)
except Exception:
    pass
layout["4.33_my"] = [112, 0, 177, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 57, 69)
    canvas.paste(_c8, (246, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 0, 303, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 100, 67)
    canvas.paste(_c9, (1214, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1214, 0, 1314, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 57, 68)
    canvas.paste(_c10, (1316, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1316, 0, 1373, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/11_icon_More.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1116, 108), _c11)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 72, 69)
    canvas.paste(_c12, (305, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [305, 0, 377, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/13_icon_Ticket_sales_end_soon.png
try:
    _c13 = get_crop(13, 547, 84)
    canvas.paste(_c13, (40, 753), _c13)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/14_icon_SHIPYARD.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1116, 108), _c14)
except Exception:
    pass
layout["SHIPYARD"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/15_icon_Show_map.png
try:
    _c15 = get_crop(15, 226, 144)
    canvas.paste(_c15, (1166, 2550), _c15)
except Exception:
    pass
layout["Show_map"] = [1166, 2550, 1392, 2694]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/16_text_Saturday_April_27.png
try:
    _c16 = get_crop(16, 451, 77)
    canvas.paste(_c16, (38, 885), _c16)
except Exception:
    pass
layout["Saturday;_April_27"] = [38, 885, 489, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/17_text_1I_00AM.png
try:
    _c17 = get_crop(17, 241, 56)
    canvas.paste(_c17, (523, 893), _c17)
except Exception:
    pass
layout["1I:00AM"] = [523, 893, 764, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/18_text_Shipyard_Open_Studios.png
try:
    _c18 = get_crop(18, 558, 144)
    canvas.paste(_c18, (288, 1155), _c18)
except Exception:
    pass
layout["Shipyard_Open_Studios"] = [288, 1155, 846, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/19_text_Spring_2024.png
try:
    _c19 = get_crop(19, 398, 144)
    canvas.paste(_c19, (946, 1195), _c19)
except Exception:
    pass
layout["Spring_2024"] = [946, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/20_text_TRUST.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (96, 1194), _c20)
except Exception:
    pass
layout["TRUST"] = [96, 1194, 240, 1338]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/21_text_Shipyard_Trust_for_the_Arts.png
try:
    _c21 = get_crop(21, 558, 144)
    canvas.paste(_c21, (288, 1155), _c21)
except Exception:
    pass
layout["Shipyard_Trust_for_the_Ar"] = [288, 1155, 846, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/22_text_OTc.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (96, 1194), _c22)
except Exception:
    pass
layout["OTc"] = [96, 1194, 240, 1338]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/23_text_934_Followers.png
try:
    _c23 = get_crop(23, 558, 144)
    canvas.paste(_c23, (288, 1155), _c23)
except Exception:
    pass
layout["934_Followers"] = [288, 1155, 846, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/24_text_Hunters_Point_Shipyard.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1422), _c24)
except Exception:
    pass
layout["Hunters_Point_Shipyard"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/25_text_days_7_hrs.png
try:
    _c25 = get_crop(25, 228, 63)
    canvas.paste(_c25, (172, 1577), _c25)
except Exception:
    pass
layout["days_7_hrs"] = [172, 1577, 400, 1640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/26_text_Refund_policy.png
try:
    _c26 = get_crop(26, 299, 63)
    canvas.paste(_c26, (138, 1685), _c26)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/27_text_The_organizer_will_review_refund_request.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1422), _c27)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/28_text_About_this_event.png
try:
    _c28 = get_crop(28, 453, 65)
    canvas.paste(_c28, (44, 1982), _c28)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 497, 2047]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/29_text_Location.png
try:
    _c29 = get_crop(29, 246, 63)
    canvas.paste(_c29, (41, 2594), _c29)
except Exception:
    pass
layout["Location"] = [41, 2594, 287, 2657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_10_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-12/30_text_S0_-_25.png
try:
    _c30 = get_crop(30, 198, 61)
    canvas.paste(_c30, (89, 2811), _c30)
except Exception:
    pass
layout["S0_-_$25"] = [89, 2811, 287, 2872]
