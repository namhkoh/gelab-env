# page_id: page_eventbrite_80babd0c0d294f03bca3ebb1c46a8a8d_06
# screenshot: 2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8.png
# step_index: 6/8
# task: Open Eventbrite. Search Art event in New York. Select the second one. Record its location and time in Google Keep Notes. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structure for mobile UI (Event page)
# Uses provided variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
status_bar_color = (189, 189, 189)      # light grey for status bar
banner_yellow = (250, 200, 25)          # bright yellow banner
content_bg = (255, 255, 255)            # white content background
card_bg = (246, 243, 247)               # very light lavender/grey for cards
muted_divider = (235, 232, 238)         # subtle divider color
shadow_color = (220, 216, 223)          # subtle shadow
page_bg = (252, 251, 253)               # warm off-white overall

# Fill overall page background
draw.rectangle([(0,0),(w,h)], fill=page_bg)

# Status bar area (top ~56px)
status_h = 56
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_color)
# slight darker top line
draw.line([(0,status_h-1),(w,status_h-1)], fill=(170,170,170), width=1)

# Large banner image area (image will be pasted on top). Draw dominant yellow background behind it.
banner_top = status_h
banner_bottom = 520
draw.rectangle([(0,banner_top),(w,banner_bottom)], fill=banner_yellow)

# Subtle vertical edge bands to mimic artwork margins (keeps it purely background)
edge_band_w = 72
edge_color = (245, 185, 30)
draw.rectangle([(0,banner_top),(edge_band_w,banner_bottom)], fill=edge_color)
draw.rectangle([(w-edge_band_w,banner_top),(w,banner_bottom)], fill=edge_color)

# White rounded content card overlapping banner (rounded top corners)
content_top = banner_bottom - 40  # overlap for rounded effect
content_radius = 34
draw.rounded_rectangle([(0,content_top),(w,h)], radius=content_radius, fill=content_bg)

# subtle separator/shadow at top of content area
draw.line([(24,content_top+2),(w-24,content_top+2)], fill=shadow_color, width=2)

# Organizer card (rounded rect) - background only, not the avatar or follow button
org_card_x0 = 48
org_card_x1 = w - 48
org_card_top = 1200
org_card_h = 160
org_card_bottom = org_card_top + org_card_h
org_card_radius = 22
draw.rounded_rectangle([(org_card_x0, org_card_top),(org_card_x1, org_card_bottom)],
                       radius=org_card_radius, fill=card_bg)
# light border around organizer card
draw.rounded_rectangle([(org_card_x0, org_card_top),(org_card_x1, org_card_bottom)],
                       radius=org_card_radius, outline=muted_divider, width=1)

# Small inner separator line inside organizer card (to visually separate text area from button area)
sep_x = org_card_x1 - 360  # leave space where Follow button will be pasted (do not draw the button)
draw.line([(sep_x, org_card_top+16),(sep_x, org_card_bottom-16)], fill=muted_divider, width=1)

# Additional subtle card shadow under organizer card
shadow_rect = [(org_card_x0+6, org_card_bottom+4),(org_card_x1-6, org_card_bottom+10)]
draw.rectangle(shadow_rect, fill=shadow_color)

# Draw small rounded pill background for a potential section header area (not the tag pills themselves)
# Place it above the organizer card (this is a background strip for badges area)
badges_bg_y0 = 640
badges_bg_y1 = 720
draw.rectangle([(48, badges_bg_y0),(w-48, badges_bg_y1)], fill=(255,255,255,0))
# top divider above badges
draw.line([(48,badges_bg_y0),(w-48,badges_bg_y0)], fill=muted_divider, width=1)

# Section separators throughout the content area (do not draw text)
sep_positions = [org_card_bottom + 40, 1520, 1840, 2180]
for y in sep_positions:
    draw.line([(48,y),(w-48,y)], fill=muted_divider, width=2)

# "About this event" header area - subtle emphasis bar (background only)
about_top = 1700
about_height = 120
draw.rectangle([(48, about_top),(w-48, about_top+about_height)], fill=content_bg)
# small accent left border under header
draw.line([(48, about_top+8),(48+6, about_top+8)], fill=(120,64,120), width=6)

# Location/Details icons area separators (left column icons will be pasted on top)
# draw light vertical guide (purely structural, very subtle)
draw.line([(140, about_top+about_height+20),(140, h-300)], fill=(245,244,246), width=1)

# Bottom sticky ticket bar background (leave space for actual button to be pasted)
sticky_top = h - 200
sticky_bottom = h
draw.rectangle([(0, sticky_top),(w, sticky_bottom)], fill=(249,249,250))
# top divider for sticky bar
draw.line([(24, sticky_top),(w-24, sticky_top)], fill=muted_divider, width=1)

# Draw left-side price area background inside sticky bar (so button on right won't be duplicated)
price_box_margin = 24
price_box = [(price_box_margin, sticky_top + 20), (w//2 - 20, sticky_bottom - 20)]
draw.rounded_rectangle(price_box, radius=12, fill=(255,255,255))
draw.line([(price_box[1][0]+6, sticky_top+18),(price_box[1][0]+6, sticky_bottom-18)], fill=muted_divider, width=1)

# Small decorative "deal" chip background near right edge but offset so not to duplicate the actual deal badge
chip_w, chip_h = 220, 64
chip_x = w - chip_w - 160
chip_y = sticky_top - 44
draw.rounded_rectangle([(chip_x, chip_y),(chip_x+chip_w, chip_y+chip_h)], radius=18, fill=(255,238,210))
draw.line([(chip_x, chip_y+chip_h),(chip_x+chip_w, chip_y+chip_h)], fill=muted_divider, width=1)

# Final subtle vertical rhythm lines to define content columns (purely structural)
left_col_x = 48
mid_col_x = 360
right_col_x = w - 48
for yy in range(content_top+40, sticky_top-40, 220):
    draw.line([(mid_col_x, yy),(mid_col_x, yy+110)], fill=(245,244,246), width=1)

# Done drawing structural backgrounds and separators.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/02_icon_Early_bird_discount.png
try:
    _c2 = get_crop(2, 449, 144)
    canvas.paste(_c2, (48, 724), _c2)
except Exception:
    pass
layout["Early_bird_discount"] = [48, 724, 497, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/03_icon_Ticket_sales_end_soon.png
try:
    _c3 = get_crop(3, 550, 84)
    canvas.paste(_c3, (502, 753), _c3)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [502, 753, 1052, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/04_icon_More.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1116, 108), _c4)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/05_icon_Performing_Visual_Arts_._Comedy.png
try:
    _c5 = get_crop(5, 234, 144)
    canvas.paste(_c5, (48, 2427), _c5)
except Exception:
    pass
layout["Performing_&_Visual_Arts_"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/06_icon_9.26.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 108), _c6)
except Exception:
    pass
layout["9.26"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/07_icon_Share.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 108), _c7)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/08_icon_2_for_1_deal.png
try:
    _c8 = get_crop(8, 570, 144)
    canvas.paste(_c8, (822, 2768), _c8)
except Exception:
    pass
layout["2_for_1_deal"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/09_icon_The_best_comedy_show_in_the_East_Village.png
try:
    _c9 = get_crop(9, 234, 144)
    canvas.paste(_c9, (48, 2427), _c9)
except Exception:
    pass
layout["The_best_comedy_show_in_t"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/10_icon_Good_Mood_Comedy.png
try:
    _c10 = get_crop(10, 441, 144)
    canvas.paste(_c10, (288, 1250), _c10)
except Exception:
    pass
layout["Good_Mood_Comedy"] = [288, 1250, 729, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/11_icon_Ticket_sales_end_soon.png
try:
    _c11 = get_crop(11, 449, 144)
    canvas.paste(_c11, (48, 724), _c11)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [48, 724, 497, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 40, 59)
    canvas.paste(_c12, (1331, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1331, 4, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 98, 60)
    canvas.paste(_c13, (1217, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [1217, 4, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/14_icon_THE.png
try:
    _c14 = get_crop(14, 51, 58)
    canvas.paste(_c14, (316, 5), _c14)
except Exception:
    pass
layout["THE"] = [316, 5, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/15_text_9.26.png
try:
    _c15 = get_crop(15, 94, 43)
    canvas.paste(_c15, (20, 17), _c15)
except Exception:
    pass
layout["9.26"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/16_text_THE.png
try:
    _c16 = get_crop(16, 117, 63)
    canvas.paste(_c16, (378, 103), _c16)
except Exception:
    pass
layout["THE"] = [378, 103, 495, 166]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/17_text_The_Good_Mood_Comedy_Show.png
try:
    _c17 = get_crop(17, 441, 144)
    canvas.paste(_c17, (288, 1250), _c17)
except Exception:
    pass
layout["The_Good_Mood_Comedy_Show"] = [288, 1250, 729, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/18_text_An_East.png
try:
    _c18 = get_crop(18, 266, 72)
    canvas.paste(_c18, (1115, 1018), _c18)
except Exception:
    pass
layout["An_East"] = [1115, 1018, 1381, 1090]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/19_text_Village_Speakeasy_Experience.png
try:
    _c19 = get_crop(19, 441, 144)
    canvas.paste(_c19, (288, 1250), _c19)
except Exception:
    pass
layout["Village_Speakeasy_Experie"] = [288, 1250, 729, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/20_text_Von.png
try:
    _c20 = get_crop(20, 89, 52)
    canvas.paste(_c20, (139, 1566), _c20)
except Exception:
    pass
layout["Von"] = [139, 1566, 228, 1618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/21_text_hrs_30_mins.png
try:
    _c21 = get_crop(21, 255, 54)
    canvas.paste(_c21, (176, 1672), _c21)
except Exception:
    pass
layout["hrs_30_mins"] = [176, 1672, 431, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/22_text_Refund_policy.png
try:
    _c22 = get_crop(22, 299, 63)
    canvas.paste(_c22, (138, 1780), _c22)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/23_text_The_organizer_will_review_refund_request.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1517), _c23)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/24_text_ZWIr.png
try:
    _c24 = get_crop(24, 165, 33)
    canvas.paste(_c24, (110, 2693), _c24)
except Exception:
    pass
layout["~ZWIr"] = [110, 2693, 275, 2726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/25_text_S0_-_8.24.png
try:
    _c25 = get_crop(25, 242, 61)
    canvas.paste(_c25, (89, 2811), _c25)
except Exception:
    pass
layout["S0_-_$8.24"] = [89, 2811, 331, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_06_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-8/26_clickable_Organizer_profile_picture.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (96, 1289), _c26)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1289, 240, 1433]
