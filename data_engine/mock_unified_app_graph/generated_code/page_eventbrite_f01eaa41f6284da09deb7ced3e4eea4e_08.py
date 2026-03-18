# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_08
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10.png
# step_index: 8/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the mobile UI
W, H = 1440, 2960

# Colors
status_bar_color = "#BFBFC3"
page_bg = "#FFFFFF"
header_divider = "#E9E8EB"
muted_divider = "#ECECEC"
card_peach = "#FFF2F1"
card_blue = "#F3F6F9"
accent_peach = "#F4A089"
accent_blue = "#2E4A7A"
ticket_border = "#345BF0"
shadow_color = "#E9E9EA"
reserve_shadow = "#E6E6E6"

# 1) Overall background (canvas already white, but fill to ensure consistent color)
draw.rectangle([(0, 0), (W, H)], fill=page_bg)

# 2) Status bar area (~72px high)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# 3) Header area (toolbar)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (W, header_bottom)], fill=page_bg)
# header bottom divider
draw.line([(32, header_bottom), (W-32, header_bottom)], fill=header_divider, width=2)

# 4) Thin divider under the short description area (approx where "Read more" area ends)
desc_divider_y = 420
draw.line([(32, desc_divider_y), (W-32, desc_divider_y)], fill=muted_divider, width=1)

# 5) Agenda section separators (top and bottom subtle lines)
agenda_top_div = 720
draw.line([(32, agenda_top_div), (W-32, agenda_top_div)], fill=muted_divider, width=1)

# 6) Agenda cards (rounded rectangles with left accent stripes)
card_x0 = 48
card_x1 = W - 48
# First card (peach)
card1_y0 = 760
card1_y1 = 920
draw.rounded_rectangle(
    [(card_x0, card1_y0), (card_x1, card1_y1)],
    radius=20,
    fill=card_peach
)
# left accent stripe for card1
stripe_margin = 20
stripe_width = 8
draw.rectangle(
    [(card_x0 + stripe_margin, card1_y0 + 20), (card_x0 + stripe_margin + stripe_width, card1_y1 - 20)],
    fill=accent_peach
)

# Second card (pale blue)
card2_y0 = 960
card2_y1 = 1100
draw.rounded_rectangle(
    [(card_x0, card2_y0), (card_x1, card2_y1)],
    radius=20,
    fill=card_blue
)
# left accent stripe for card2
draw.rectangle(
    [(card_x0 + stripe_margin, card2_y0 + 20), (card_x0 + stripe_margin + stripe_width, card2_y1 - 20)],
    fill=accent_blue
)

# 7) Separator line between agenda and location area
sep_y = 1150
draw.line([(32, sep_y), (W-32, sep_y)], fill=muted_divider, width=2)

# 8) Location area divider (above the map/show map area)
loc_divider_y = 1520
draw.line([(32, loc_divider_y), (W-32, loc_divider_y)], fill=muted_divider, width=1)

# 9) Subtle horizontal divider further down to suggest section breaks
draw.line([(32, 2000), (W-32, 2000)], fill=muted_divider, width=1)

# 10) Ticket selection card (rounded, bordered) above the reserve button
ticket_x0 = 72
ticket_x1 = W - 72
ticket_y1 = 2660  # bottom of ticket card
ticket_y0 = ticket_y1 - 300  # height ~300
# subtle shadow behind ticket card
shadow_offset = 8
draw.rounded_rectangle(
    [(ticket_x0 + shadow_offset, ticket_y0 + shadow_offset), (ticket_x1 + shadow_offset, ticket_y1 + shadow_offset)],
    radius=24,
    fill=shadow_color
)
# main ticket card (white fill with blue border)
draw.rounded_rectangle(
    [(ticket_x0, ticket_y0), (ticket_x1, ticket_y1)],
    radius=24,
    fill=page_bg,
    outline=ticket_border,
    width=8
)

# 11) Small dividing line inside the ticket card to separate content area (visual structure only)
inner_divide_x = ticket_x0 + 880
draw.line([(inner_divide_x, ticket_y0 + 24), (inner_divide_x, ticket_y1 - 24)], fill=muted_divider, width=1)

# 12) Reserve button area (leave blank but draw subtle shadow to indicate its placement)
reserve_top = 2756
reserve_left = 72
reserve_right = W - 72
reserve_bottom = reserve_top + 132
# light shadow behind the reserve button (do not draw the button itself since it's a detected element)
draw.rectangle([(reserve_left, reserve_top - 8), (reserve_right, reserve_bottom + 8)], fill=reserve_shadow)

# 13) Final subtle bottom safe-area fill (to blend edge)
draw.rectangle([(0, H-40), (W, H)], fill=page_bg)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/01_icon_Sports_Basement_Berkeley_Community.png
try:
    _c1 = get_crop(1, 234, 144)
    canvas.paste(_c1, (48, 371), _c1)
except Exception:
    pass
layout["Sports_Basement_Berkeley_"] = [48, 371, 282, 515]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1296, 132)
    canvas.paste(_c3, (72, 2756), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/04_icon_Decrease.png
try:
    _c4 = get_crop(4, 99, 96)
    canvas.paste(_c4, (996, 2444), _c4)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 93, 105)
    canvas.paste(_c6, (1108, 2441), _c6)
except Exception:
    pass
layout["icon_6"] = [1108, 2441, 1201, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 56, 57)
    canvas.paste(_c7, (183, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [183, 4, 239, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 58, 56)
    canvas.paste(_c8, (311, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [311, 6, 369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/09_icon_4.36.png
try:
    _c9 = get_crop(9, 56, 55)
    canvas.paste(_c9, (116, 7), _c9)
except Exception:
    pass
layout["4.36"] = [116, 7, 172, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 55)
    canvas.paste(_c10, (248, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [248, 6, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 99, 59)
    canvas.paste(_c11, (1216, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1216, 3, 1315, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/12_icon_Sports_Basement_Berkeley_2727_Milvia_Str.png
try:
    _c12 = get_crop(12, 226, 144)
    canvas.paste(_c12, (1166, 1709), _c12)
except Exception:
    pass
layout["Sports_Basement_Berkeley;"] = [1166, 1709, 1392, 1853]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/13_icon_Free.png
try:
    _c13 = get_crop(13, 75, 72)
    canvas.paste(_c13, (249, 2588), _c13)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/14_icon_Free.png
try:
    _c14 = get_crop(14, 138, 102)
    canvas.paste(_c14, (98, 2573), _c14)
except Exception:
    pass
layout["Free"] = [98, 2573, 236, 2675]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/15_icon_Show_map.png
try:
    _c15 = get_crop(15, 226, 144)
    canvas.paste(_c15, (1166, 1709), _c15)
except Exception:
    pass
layout["Show_map"] = [1166, 1709, 1392, 1853]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 44, 57)
    canvas.paste(_c16, (1326, 4), _c16)
except Exception:
    pass
layout["icon_16"] = [1326, 4, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/17_icon_Across_the_Street_Hands_on_Experience.png
try:
    _c17 = get_crop(17, 226, 144)
    canvas.paste(_c17, (1166, 1709), _c17)
except Exception:
    pass
layout["Across_the_Street:_Hands_"] = [1166, 1709, 1392, 1853]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/18_icon_4.36.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (36, 108), _c18)
except Exception:
    pass
layout["4.36"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 46, 55)
    canvas.paste(_c19, (384, 7), _c19)
except Exception:
    pass
layout["icon_19"] = [384, 7, 430, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/20_text_4.36.png
try:
    _c20 = get_crop(20, 89, 45)
    canvas.paste(_c20, (22, 15), _c20)
except Exception:
    pass
layout["4.36"] = [22, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/21_text_Backpacking_Clinic_W.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (36, 108), _c21)
except Exception:
    pass
layout["Backpacking_Clinic_W=_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/22_text_Intro_to_Backpacking_clinic_with_Sports_.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1116, 108), _c22)
except Exception:
    pass
layout["Intro_to_Backpacking_clin"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/23_text_Read_more.png
try:
    _c23 = get_crop(23, 234, 144)
    canvas.paste(_c23, (48, 371), _c23)
except Exception:
    pass
layout["Read_more"] = [48, 371, 282, 515]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/24_text_Agenda.png
try:
    _c24 = get_crop(24, 229, 80)
    canvas.paste(_c24, (41, 631), _c24)
except Exception:
    pass
layout["Agenda"] = [41, 631, 270, 711]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/25_text_Location.png
try:
    _c25 = get_crop(25, 244, 61)
    canvas.paste(_c25, (43, 1754), _c25)
except Exception:
    pass
layout["Location"] = [43, 1754, 287, 1815]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_08_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-10/26_text_General_Admission.png
try:
    _c26 = get_crop(26, 75, 72)
    canvas.paste(_c26, (249, 2588), _c26)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]
