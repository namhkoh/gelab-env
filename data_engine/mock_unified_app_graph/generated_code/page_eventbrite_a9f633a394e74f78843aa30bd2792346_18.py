# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_18
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20.png
# step_index: 18/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960). Fonts: font_sm, font_md, font_lg, font_xl
# Draw overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FBFAFF")

# Status bar area (top)
STATUS_H = 72
draw.rectangle((0, 0, 1440, STATUS_H), fill="#DAD8D6")

# Top banner / hero image area (do not draw any icons or small circular backgrounds)
BANNER_Y0 = STATUS_H
BANNER_Y1 = 460
# Left tile
draw.rectangle((0, BANNER_Y0, 720, BANNER_Y1), fill="#E8D6CC")
# Middle tile
draw.rectangle((720, BANNER_Y0, 1120, BANNER_Y1), fill="#F3EADD")
# Right tile
draw.rectangle((1120, BANNER_Y0, 1440, BANNER_Y1), fill="#F0DCC6")
# Subtle divider/shadow beneath banner
draw.rectangle((0, BANNER_Y1 - 4, 1440, BANNER_Y1), fill="#EFEFF1")

# Main content background (the page is mostly white, keep it clean)
CONTENT_Y0 = BANNER_Y1
draw.rectangle((0, CONTENT_Y0, 1440, 2324), fill="#FFFFFF")

# Thin faint horizontal rule under top area
draw.line((48, CONTENT_Y0 + 12, 1392, CONTENT_Y0 + 12), fill="#F0EDF4", width=1)

# "Going fast" / badges area is made of small pill elements in the screenshot (these will be pasted),
# so do NOT draw those pills. Instead, provide subtle background spacing only:
BADGE_AREA_Y = BANNER_Y1 + 240
draw.rectangle((48, BADGE_AREA_Y - 18, 1392, BADGE_AREA_Y + 18), fill="#FFFFFF")

# Organizer card (rounded rectangle background behind organizer info & Follow button)
# This is intentionally a soft, pale card background to match screenshot.
ORG_X0, ORG_Y0 = 48, 1100
ORG_X1, ORG_Y1 = 1392, 1260
draw.rounded_rectangle((ORG_X0, ORG_Y0, ORG_X1, ORG_Y1),
                       radius=20,
                       fill="#F7F6FB",
                       outline="#E6E4F2",
                       width=2)

# Subtle inner divider on organizer card to suggest separation between text & follow button (no icons)
DIV_X = ORG_X1 - 320
draw.line((DIV_X, ORG_Y0 + 18, DIV_X, ORG_Y1 - 18), fill="#F0EDF4", width=1)

# Info rows area (location, duration, refund policy) - draw subtle separators and spacing
INFO_START_Y = ORG_Y1 + 40
# Draw small icon placeholder backgrounds? NO icons must not be drawn. Only separators:
draw.line((48, INFO_START_Y + 140, 1392, INFO_START_Y + 140), fill="#F1EFF5", width=1)

# Horizontal rule separating main content and "About this event" area
ABOUT_DIV_Y = INFO_START_Y + 220
draw.line((48, ABOUT_DIV_Y, 1392, ABOUT_DIV_Y), fill="#EDECF1", width=1)

# "About this event" section background spacing (subtle off-white block)
ABOUT_Y0 = ABOUT_DIV_Y + 24
ABOUT_Y1 = ABOUT_Y0 + 220
draw.rectangle((48, ABOUT_Y0, 1392, ABOUT_Y1), fill="#FFFFFF")

# Light rounded container for details further down (but avoid overlapping bottom CTA area)
DETAILS_X0, DETAILS_Y0 = 48, ABOUT_Y1 + 28
DETAILS_X1, DETAILS_Y1 = 1392, 2160
draw.rounded_rectangle((DETAILS_X0, DETAILS_Y0, DETAILS_X1, DETAILS_Y1),
                       radius=12,
                       fill="#FFFFFF",
                       outline="#ECEAF2",
                       width=2)

# Large subtle divider above the ticket/reserve area (do not draw the reserve CTA itself)
TICKET_DIV_Y = 2168
draw.line((48, TICKET_DIV_Y, 1392, TICKET_DIV_Y), fill="#EDEAF0", width=2)

# Footer area above the reserved-spot CTA (keep blank/light)
draw.rectangle((0, 2324 - 120, 1440, 2324), fill="#FFFFFF")

# Add a faint left shadow line to anchor content column
draw.line((48, BANNER_Y1 + 6, 48, 2324 - 6), fill="#F3F1F6", width=2)

# Small decorative corner radii shadows for the main white content frame (very subtle)
draw.line((48 + 6, 2324 - 6, 1392 - 6, 2324 - 6), fill="#F5F4F8", width=1)
draw.line((1392 - 6, ORG_Y0 + 6, 1392 - 6, 2324 - 6), fill="#F5F4F8", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 112, 106)
    canvas.paste(_c1, (988, 2440), _c1)
except Exception:
    pass
layout["icon_1"] = [988, 2440, 1100, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 107, 104)
    canvas.paste(_c2, (1215, 2442), _c2)
except Exception:
    pass
layout["icon_2"] = [1215, 2442, 1322, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 93, 104)
    canvas.paste(_c3, (1108, 2441), _c3)
except Exception:
    pass
layout["icon_3"] = [1108, 2441, 1201, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/04_icon_4.52.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["4.52"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/05_icon_More.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/06_icon_Reserve_a_spot.png
try:
    _c6 = get_crop(6, 1440, 636)
    canvas.paste(_c6, (0, 2324), _c6)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/07_icon_Going_fast.png
try:
    _c7 = get_crop(7, 334, 86)
    canvas.paste(_c7, (41, 753), _c7)
except Exception:
    pass
layout["Going_fast"] = [41, 753, 375, 839]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 47, 68)
    canvas.paste(_c8, (1156, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1156, 1, 1203, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/09_icon_Share.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1260, 108), _c9)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/10_icon_Los_Angeles_Food_Policy_Council.png
try:
    _c10 = get_crop(10, 684, 144)
    canvas.paste(_c10, (144, 1155), _c10)
except Exception:
    pass
layout["Los_Angeles_Food_Policy_C"] = [144, 1155, 828, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/11_icon_4.52.png
try:
    _c11 = get_crop(11, 59, 63)
    canvas.paste(_c11, (182, 1), _c11)
except Exception:
    pass
layout["4.52"] = [182, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 63)
    canvas.paste(_c12, (1327, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1327, 3, 1372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 61)
    canvas.paste(_c13, (248, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [248, 2, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/14_icon_Ticket_sales_end_soon.png
try:
    _c14 = get_crop(14, 684, 144)
    canvas.paste(_c14, (144, 1155), _c14)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [144, 1155, 828, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/15_icon_4.52.png
try:
    _c15 = get_crop(15, 59, 64)
    canvas.paste(_c15, (115, 1), _c15)
except Exception:
    pass
layout["4.52"] = [115, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/16_icon_Hosted_by_Healthy_Neighborhood_Market_Ne.png
try:
    _c16 = get_crop(16, 234, 107)
    canvas.paste(_c16, (48, 2217), _c16)
except Exception:
    pass
layout["Hosted_by_Healthy_Neighbo"] = [48, 2217, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 59, 60)
    canvas.paste(_c17, (312, 3), _c17)
except Exception:
    pass
layout["icon_17"] = [312, 3, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/18_icon_Ticket_sales_end_soon.png
try:
    _c18 = get_crop(18, 549, 84)
    canvas.paste(_c18, (378, 753), _c18)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [378, 753, 927, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 101, 65)
    canvas.paste(_c19, (1214, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [1214, 2, 1315, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 50, 60)
    canvas.paste(_c20, (383, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [383, 3, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/21_icon_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1422), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/22_text_4.52.png
try:
    _c22 = get_crop(22, 89, 43)
    canvas.paste(_c22, (22, 17), _c22)
except Exception:
    pass
layout["4.52"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/23_text_Product_Photography_Workshop.png
try:
    _c23 = get_crop(23, 684, 144)
    canvas.paste(_c23, (144, 1155), _c23)
except Exception:
    pass
layout["Product_Photography_Works"] = [144, 1155, 828, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/24_text_General_Admission.png
try:
    _c24 = get_crop(24, 234, 107)
    canvas.paste(_c24, (48, 2217), _c24)
except Exception:
    pass
layout["General_Admission"] = [48, 2217, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_18_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-20/25_text_Free.png
try:
    _c25 = get_crop(25, 105, 48)
    canvas.paste(_c25, (116, 2599), _c25)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]
