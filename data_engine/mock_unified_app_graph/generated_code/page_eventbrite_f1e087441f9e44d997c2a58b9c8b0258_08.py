# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_08
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10.png
# step_index: 8/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background/base colors
bg_color = (250,249,252)        # very light off-white background
status_bar_color = (190,190,190)  # muted grey status bar
banner_top_color = (12,29,50)     # dark navy for banner top
banner_bottom_color = (47,123,172)  # blue for banner bottom (subtle gradient)
card_bg = (247,245,250)           # pale card background (very light purple/grey)
pill_bg = (240,232,255)           # pale purple pill for "ticket sales" etc.
muted_divider = (230,228,235)     # subtle divider lines
bottom_bar_bg = (249,248,250)     # slightly different pale background for sticky footer
cta_bg = (203,57,18)              # orange for behind "Get tickets" area
shadow_color = (0,0,0,20)

w, h = canvas.size

# Fill main background
draw.rectangle([(0,0),(w,h)], fill=bg_color)

# Status bar (top ~50px)
status_h = 84
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_color)

# Header/banner area below status bar
banner_top = status_h
banner_height = 480
banner_bottom = banner_top + banner_height

# Vertical gradient for banner
for i in range(banner_height):
    ratio = i / max(1, banner_height - 1)
    r = int(banner_top_color[0] * (1-ratio) + banner_bottom_color[0] * ratio)
    g = int(banner_top_color[1] * (1-ratio) + banner_bottom_color[1] * ratio)
    b = int(banner_top_color[2] * (1-ratio) + banner_bottom_color[2] * ratio)
    draw.line([(0, banner_top + i), (w, banner_top + i)], fill=(r,g,b))

# Slight dark overlay at bottom of banner to simulate image fade (content will be pasted on top)
overlay_height = 90
overlay_box = (0, banner_bottom - overlay_height, w, banner_bottom)
draw.rectangle(overlay_box, fill=(0,0,0,40))

# Horizontal divider under banner
draw.line([(40, banner_bottom + 12), (w-40, banner_bottom + 12)], fill=muted_divider, width=1)

# Ticket-sales pill/background (rounded)
pill_left = 40
pill_top = banner_bottom + 96
pill_right = 600
pill_bottom = pill_top + 72
draw.rounded_rectangle([(pill_left, pill_top), (pill_right, pill_bottom)], radius=36, fill=pill_bg)

# Large content area separation: main whitespace area (we keep canvas bg, but add subtle separator lines)
section_y = pill_bottom + 40
# Subtle top margin line
draw.line([(40, section_y), (w-40, section_y)], fill=muted_divider, width=1)

# Organizer card (rounded rectangle behind organizer row with follow button)
card_left = 40
card_top = 1150
card_right = w - 40
card_bottom = card_top + 148
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)], radius=24, fill=card_bg)

# Add subtle inner divider line within organizer card (to visually separate left content and right button area)
inner_line_x = card_right - 260
draw.line([(inner_line_x, card_top + 16), (inner_line_x, card_bottom - 16)], fill=muted_divider, width=1)

# Small shadow under the organizer card for depth (soft line)
draw.line([(card_left, card_bottom+2),(card_right, card_bottom+2)], fill=(220,218,230), width=1)

# Section list separators (icons/text will be pasted on top)
list_start = card_bottom + 36
# Draw three subtle dividers spaced to match the "location/time/refund" list layout
draw.line([(40, list_start + 200), (w-40, list_start + 200)], fill=muted_divider, width=1)
draw.line([(40, list_start + 380), (w-40, list_start + 380)], fill=muted_divider, width=1)

# "About this event" divider and spacing (header will be pasted)
about_div_y = 1960
draw.line([(40, about_div_y), (w-40, about_div_y)], fill=muted_divider, width=1)

# Tag pills area for categories (pale rounded pills)
tag1_box = (48, 2332, 320, 2366)
tag2_box = (336, 2332, 520, 2366)
draw.rounded_rectangle([(tag1_box[0], tag1_box[1]), (tag1_box[2], tag1_box[3])], radius=20, fill=(242,243,247))
draw.rounded_rectangle([(tag2_box[0], tag2_box[1]), (tag2_box[2], tag2_box[3])], radius=20, fill=(242,243,247))

# Subtle divider above Location section
location_div_y = 2536
draw.line([(40, location_div_y), (w-40, location_div_y)], fill=muted_divider, width=1)

# Location header area: leave space; draw thin underline to denote end of section
draw.line([(40, 2660), (w-40, 2660)], fill=muted_divider, width=1)

# Bottom sticky footer bar (background)
footer_top = 2680
draw.rectangle([(0, footer_top), (w, h)], fill=bottom_bar_bg)

# Thin divider above footer
draw.line([(0, footer_top), (w, footer_top)], fill=muted_divider, width=1)

# CTA background (behind Get tickets button). We only draw the colored rectangle area, not any text or icon.
cta_left = 820
cta_top = 2756
cta_right = w - 48
cta_bottom = cta_top + 108
draw.rounded_rectangle([(cta_left, cta_top), (cta_right, cta_bottom)], radius=12, fill=cta_bg)

# Price area left side (subtle placeholder box behind price text)
price_box_left = 48
price_box_top = 2768
price_box_right = 420
price_box_bottom = price_box_top + 72
draw.rectangle([(price_box_left, price_box_top), (price_box_right, price_box_bottom)], fill=bottom_bar_bg)

# Final subtle horizontal separators for visual rhythm
draw.line([(40, 1520), (w-40, 1520)], fill=muted_divider, width=1)
draw.line([(40, 1840), (w-40, 1840)], fill=muted_divider, width=1)

# Note: All UI text and icons are intentionally not drawn here; they will be pasted on top separately.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/00_icon_Get_tickets.png
try:
    _c0 = get_crop(0, 570, 144)
    canvas.paste(_c0, (822, 2768), _c0)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/01_icon_Follow.png
try:
    _c1 = get_crop(1, 331, 144)
    canvas.paste(_c1, (1013, 1195), _c1)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/02_icon_4.33_my.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["4.33_my"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/03_icon_More.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/05_icon_Celebrate_Art_at_San_Francisco_s_Premier.png
try:
    _c5 = get_crop(5, 234, 144)
    canvas.paste(_c5, (48, 2332), _c5)
except Exception:
    pass
layout["Celebrate_Art_at_San_Fran"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/06_icon_Performing_Visual_Arts.png
try:
    _c6 = get_crop(6, 234, 144)
    canvas.paste(_c6, (48, 2332), _c6)
except Exception:
    pass
layout["Performing_&_Visual_Arts"] = [48, 2332, 282, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/07_icon_4.33_my.png
try:
    _c7 = get_crop(7, 65, 70)
    canvas.paste(_c7, (178, 0), _c7)
except Exception:
    pass
layout["4.33_my"] = [178, 0, 243, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/08_icon_4.33_my.png
try:
    _c8 = get_crop(8, 65, 70)
    canvas.paste(_c8, (112, 0), _c8)
except Exception:
    pass
layout["4.33_my"] = [112, 0, 177, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 57, 69)
    canvas.paste(_c9, (246, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [246, 0, 303, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 100, 67)
    canvas.paste(_c10, (1214, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1214, 0, 1314, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 72, 68)
    canvas.paste(_c11, (305, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [305, 0, 377, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 68)
    canvas.paste(_c12, (1316, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1316, 0, 1373, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/13_icon_Ticket_sales_end_soon.png
try:
    _c13 = get_crop(13, 547, 84)
    canvas.paste(_c13, (40, 753), _c13)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/14_icon_SHIPYARD.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1116, 108), _c14)
except Exception:
    pass
layout["SHIPYARD"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/15_icon_Show_map.png
try:
    _c15 = get_crop(15, 226, 144)
    canvas.paste(_c15, (1166, 2550), _c15)
except Exception:
    pass
layout["Show_map"] = [1166, 2550, 1392, 2694]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/16_icon_Shipyard_Trust_for_the_Arts.png
try:
    _c16 = get_crop(16, 558, 144)
    canvas.paste(_c16, (288, 1155), _c16)
except Exception:
    pass
layout["Shipyard_Trust_for_the_Ar"] = [288, 1155, 846, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/17_text_Saturday_April_27.png
try:
    _c17 = get_crop(17, 451, 77)
    canvas.paste(_c17, (38, 885), _c17)
except Exception:
    pass
layout["Saturday;_April_27"] = [38, 885, 489, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/18_text_1I_00AM.png
try:
    _c18 = get_crop(18, 241, 56)
    canvas.paste(_c18, (523, 893), _c18)
except Exception:
    pass
layout["1I:00AM"] = [523, 893, 764, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/19_text_Shipyard_Open_Studios.png
try:
    _c19 = get_crop(19, 558, 144)
    canvas.paste(_c19, (288, 1155), _c19)
except Exception:
    pass
layout["Shipyard_Open_Studios"] = [288, 1155, 846, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/20_text_Spring_2024.png
try:
    _c20 = get_crop(20, 331, 144)
    canvas.paste(_c20, (1013, 1195), _c20)
except Exception:
    pass
layout["Spring_2024"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/21_text_Hunters_Point_Shipyard.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1422), _c21)
except Exception:
    pass
layout["Hunters_Point_Shipyard"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/22_text_days_7_hrs.png
try:
    _c22 = get_crop(22, 228, 63)
    canvas.paste(_c22, (172, 1577), _c22)
except Exception:
    pass
layout["days_7_hrs"] = [172, 1577, 400, 1640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/23_text_Refund_policy.png
try:
    _c23 = get_crop(23, 299, 63)
    canvas.paste(_c23, (138, 1685), _c23)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/24_text_The_organizer_will_review_refund_request.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1422), _c24)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/25_text_About_this_event.png
try:
    _c25 = get_crop(25, 453, 65)
    canvas.paste(_c25, (44, 1982), _c25)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 497, 2047]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/26_text_Location.png
try:
    _c26 = get_crop(26, 246, 63)
    canvas.paste(_c26, (41, 2594), _c26)
except Exception:
    pass
layout["Location"] = [41, 2594, 287, 2657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/27_text_S0_-_25.png
try:
    _c27 = get_crop(27, 198, 61)
    canvas.paste(_c27, (89, 2811), _c27)
except Exception:
    pass
layout["S0_-_$25"] = [89, 2811, 287, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_08_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-10/28_clickable_Organizer_profile_picture.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (96, 1194), _c28)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1194, 240, 1338]
