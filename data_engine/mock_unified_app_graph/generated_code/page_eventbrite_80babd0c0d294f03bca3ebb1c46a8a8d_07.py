# page_id: page_eventbrite_80babd0c0d294f03bca3ebb1c46a8a8d_07
# screenshot: 2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9.png
# step_index: 7/8
# task: Open Eventbrite. Search Art event in New York. Select the second one. Record its location and time in Google Keep Notes. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 84
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))

# Yellow banner gradient beneath status bar (main hero image background color)
banner_top = status_h
banner_bottom = 720
start_color = (255, 193, 7)   # deep yellow
end_color = (255, 236, 88)    # lighter yellow

for y in range(banner_top, banner_bottom):
    t = (y - banner_top) / max(1, (banner_bottom - banner_top))
    r = int(start_color[0] * (1 - t) + end_color[0] * t)
    g = int(start_color[1] * (1 - t) + end_color[1] * t)
    b = int(start_color[2] * (1 - t) + end_color[2] * t)
    draw.line([(0, y), (1440, y)], fill=(r, g, b))

# Subtle darker edges on left and right of banner to mimic photograph borders
edge_width = 80
draw.rectangle([(0, banner_top), (edge_width, banner_bottom)], fill=(255, 180, 0))
draw.rectangle([(1440 - edge_width, banner_top), (1440, banner_bottom)], fill=(255, 180, 0))

# Very subtle drop shadow under the banner
shadow_y = banner_bottom
draw.rectangle([(0, shadow_y), (1440, shadow_y + 6)], fill=(245, 245, 245))

# Organizer/host card background (rounded)
card_x0, card_x1 = 48, 1392
card_y0, card_y1 = 1220, 1404
card_radius = 28
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)],
                       radius=card_radius,
                       fill=(246, 247, 250),
                       outline=(226, 229, 236),
                       width=1)

# Thin divider under organizer card/details area
divider_y = 1500
draw.line([(48, divider_y), (1392, divider_y)], fill=(236, 237, 241), width=2)

# Secondary thin divider lower down (between sections)
divider_y2 = 1840
draw.line([(48, divider_y2), (1392, divider_y2)], fill=(238, 239, 243), width=1)

# About section subtle top/bottom spacing separators
about_top = 1680
draw.line([(48, about_top), (1392, about_top)], fill=(245, 245, 247), width=1)

# Another faint separator before the "Location" area
loc_sep = 2200
draw.line([(48, loc_sep), (1392, loc_sep)], fill=(245, 245, 247), width=1)

# Bottom action bar background (leave space for buttons/icons which will be pasted on top)
bottom_bar_top = 2760
draw.rectangle([(0, bottom_bar_top), (1440, 2960)], fill=(250, 248, 249))
# subtle top border to separate content
draw.line([(0, bottom_bar_top), (1440, bottom_bar_top)], fill=(226, 224, 225), width=2)

# Add soft drop shadow above bottom bar
draw.rectangle([(0, bottom_bar_top - 6), (1440, bottom_bar_top)], fill=(245, 244, 245))

# Small rounded panel on right side near bottom (background for ticket CTA area)
# NOTE: We draw a base background here but avoid any shapes that would replicate the actual "Get tickets" button region.
cta_bg_x0, cta_bg_x1 = 720, 1440
cta_bg_y0, cta_bg_y1 = bottom_bar_top + 12, 2960 - 12
draw.rounded_rectangle([(cta_bg_x0, cta_bg_y0), (cta_bg_x1 - 12, cta_bg_y1)],
                       radius=12,
                       fill=(255, 245, 242))

# Light content area background block for the main content (subtle off-white to set sections apart)
content_block_top = banner_bottom
content_block_bottom = bottom_bar_top
draw.rectangle([(0, content_block_top), (1440, content_block_bottom)], fill=(255, 255, 255))

# Add a very subtle large rounded card behind the "About this event" area
about_card_x0, about_card_x1 = 48, 1392
about_card_y0, about_card_y1 = 1580, 1980
draw.rounded_rectangle([(about_card_x0, about_card_y0), (about_card_x1, about_card_y1)],
                       radius=18,
                       fill=(252, 252, 253),
                       outline=(240, 240, 242),
                       width=1)

# Final soft separators for sections nearer the top of the content
draw.line([(48, 880), (1392, 880)], fill=(245, 245, 247), width=1)
draw.line([(48, 1040), (1392, 1040)], fill=(245, 245, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1290), _c0)
except Exception:
    pass
layout["Following"] = [946, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/02_icon_Early_bird_discount.png
try:
    _c2 = get_crop(2, 449, 144)
    canvas.paste(_c2, (48, 724), _c2)
except Exception:
    pass
layout["Early_bird_discount"] = [48, 724, 497, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/03_icon_More.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/04_icon_Ticket_sales_end_soon.png
try:
    _c4 = get_crop(4, 549, 84)
    canvas.paste(_c4, (503, 753), _c4)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [503, 753, 1052, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/05_icon_Performing_Visual_Arts_._Comedy.png
try:
    _c5 = get_crop(5, 234, 144)
    canvas.paste(_c5, (48, 2427), _c5)
except Exception:
    pass
layout["Performing_&_Visual_Arts_"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/06_icon_Share.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1260, 108), _c6)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/07_icon_9.26.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (36, 108), _c7)
except Exception:
    pass
layout["9.26"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/08_icon_2_for_1_deal.png
try:
    _c8 = get_crop(8, 570, 144)
    canvas.paste(_c8, (822, 2768), _c8)
except Exception:
    pass
layout["2_for_1_deal"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/09_icon_Ticket_sales_end_soon.png
try:
    _c9 = get_crop(9, 449, 144)
    canvas.paste(_c9, (48, 724), _c9)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [48, 724, 497, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/10_icon_The_best_comedy_show_in_the_East_Village.png
try:
    _c10 = get_crop(10, 234, 144)
    canvas.paste(_c10, (48, 2427), _c10)
except Exception:
    pass
layout["The_best_comedy_show_in_t"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 40, 60)
    canvas.paste(_c11, (1331, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1331, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 98, 60)
    canvas.paste(_c12, (1217, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1217, 4, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/13_icon_THE.png
try:
    _c13 = get_crop(13, 51, 58)
    canvas.paste(_c13, (316, 5), _c13)
except Exception:
    pass
layout["THE"] = [316, 5, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/14_text_9.26.png
try:
    _c14 = get_crop(14, 94, 43)
    canvas.paste(_c14, (20, 17), _c14)
except Exception:
    pass
layout["9.26"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/15_text_THE.png
try:
    _c15 = get_crop(15, 117, 63)
    canvas.paste(_c15, (378, 103), _c15)
except Exception:
    pass
layout["THE"] = [378, 103, 495, 166]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/16_text_The_Good_Mood_Comedy_Show.png
try:
    _c16 = get_crop(16, 441, 144)
    canvas.paste(_c16, (288, 1250), _c16)
except Exception:
    pass
layout["The_Good_Mood_Comedy_Show"] = [288, 1250, 729, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/17_text_An_East.png
try:
    _c17 = get_crop(17, 266, 72)
    canvas.paste(_c17, (1115, 1018), _c17)
except Exception:
    pass
layout["An_East"] = [1115, 1018, 1381, 1090]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/18_text_Village_Speakeasy_Experience.png
try:
    _c18 = get_crop(18, 441, 144)
    canvas.paste(_c18, (288, 1250), _c18)
except Exception:
    pass
layout["Village_Speakeasy_Experie"] = [288, 1250, 729, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/19_text_Good_Mood_Comedy.png
try:
    _c19 = get_crop(19, 441, 144)
    canvas.paste(_c19, (288, 1250), _c19)
except Exception:
    pass
layout["Good_Mood_Comedy"] = [288, 1250, 729, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/20_text_130_Followers.png
try:
    _c20 = get_crop(20, 441, 144)
    canvas.paste(_c20, (288, 1250), _c20)
except Exception:
    pass
layout["130_Followers"] = [288, 1250, 729, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/21_text_Von.png
try:
    _c21 = get_crop(21, 89, 52)
    canvas.paste(_c21, (139, 1566), _c21)
except Exception:
    pass
layout["Von"] = [139, 1566, 228, 1618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/22_text_hrs_30_mins.png
try:
    _c22 = get_crop(22, 255, 54)
    canvas.paste(_c22, (176, 1672), _c22)
except Exception:
    pass
layout["hrs_30_mins"] = [176, 1672, 431, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/23_text_Refund_policy.png
try:
    _c23 = get_crop(23, 299, 63)
    canvas.paste(_c23, (138, 1780), _c23)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/24_text_The_organizer_will_review_refund_request.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1517), _c24)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/25_text_ZWIr.png
try:
    _c25 = get_crop(25, 165, 33)
    canvas.paste(_c25, (110, 2693), _c25)
except Exception:
    pass
layout["~ZWIr"] = [110, 2693, 275, 2726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/26_text_S0_-_8.24.png
try:
    _c26 = get_crop(26, 242, 61)
    canvas.paste(_c26, (89, 2811), _c26)
except Exception:
    pass
layout["S0_-_$8.24"] = [89, 2811, 331, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_07_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-9/27_clickable_Organizer_profile_picture.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (96, 1289), _c27)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1289, 240, 1433]
