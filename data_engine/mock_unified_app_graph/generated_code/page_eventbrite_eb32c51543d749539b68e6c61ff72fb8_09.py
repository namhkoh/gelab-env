# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_09
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11.png
# step_index: 9/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background and structural UI elements for the mobile page.
# Uses provided 'canvas' (1440x2960 RGB) and 'draw' (ImageDraw).

# Fill background (dominant color: white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top) - light grey background
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(200, 200, 200))

# Subtle divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(185, 185, 185), width=1)

# Header / toolbar background area (under status bar)
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))

# Soft shadow under header to separate from content
# draw a couple of faint lines to mimic a shadow
draw.line((0, header_bottom, 1440, header_bottom), fill=(236, 236, 236), width=1)
draw.line((0, header_bottom+1, 1440, header_bottom+1), fill=(245, 245, 245), width=1)

# Large content card area behind the date selection (rounded rectangle)
card_x0 = 32
card_x1 = 1408
card_y0 = 220
card_y1 = 760
card_radius = 16
# very subtle off-white card to group date items
draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1),
                       radius=card_radius,
                       fill=(250, 250, 252),
                       outline=(235, 234, 238),
                       width=1)

# Section separators inside the card (leave space where text/icons will be pasted)
# Separator above the first date field
sep_y1 = 300
draw.line((card_x0 + 16, sep_y1, card_x1 - 16, sep_y1), fill=(230, 229, 233), width=1)
# Separator between start and end areas (visual guide)
sep_y2 = 420
draw.line((card_x0 + 16, sep_y2, card_x1 - 16, sep_y2), fill=(230, 229, 233), width=1)
# Separator under the clickable "Choose a date" area
sep_y3 = 560
draw.line((card_x0 + 16, sep_y3, card_x1 - 16, sep_y3), fill=(242, 241, 245), width=1)

# Subtle left accent stripe to add structure (thin)
accent_x = card_x0 + 8
draw.rectangle((accent_x, card_y0 + 12, accent_x + 6, card_y1 - 12), fill=(245, 241, 255))

# Bottom sticky area shadow (above the apply button)
bottom_shadow_y = 2756
draw.line((24, bottom_shadow_y, 1416, bottom_shadow_y), fill=(230, 229, 233), width=2)

# Draw rounded rectangle background for the bottom "Apply date range" control area.
# NOTE: the button's actual content (text/icons) will be pasted on top; we only draw the background/frame.
btn_x0 = 48
btn_y0 = 2768
btn_x1 = btn_x0 + 1344
btn_y1 = btn_y0 + 144
btn_radius = 12

# Button background (white) and border
draw.rounded_rectangle((btn_x0, btn_y0, btn_x1, btn_y1),
                       radius=btn_radius,
                       fill=(255, 255, 255),
                       outline=(160, 155, 170),
                       width=4)

# Slight inner highlight to give subtle 3D effect
inner_inset = 6
draw.rounded_rectangle((btn_x0 + inner_inset, btn_y0 + inner_inset, btn_x1 - inner_inset, btn_y1 - inner_inset),
                       radius=btn_radius - 4,
                       outline=(245, 244, 246),
                       width=1)

# Optional subtle page side margins (visual gutters)
gutters = 32
draw.line((gutters, header_bottom + 8, gutters, 2760), fill=(248, 248, 249), width=1)
draw.line((1440 - gutters, header_bottom + 8, 1440 - gutters, 2760), fill=(248, 248, 249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/01_icon_7.47.png
try:
    _c1 = get_crop(1, 56, 63)
    canvas.paste(_c1, (183, 2), _c1)
except Exception:
    pass
layout["7.47"] = [183, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/02_icon_7.47.png
try:
    _c2 = get_crop(2, 58, 63)
    canvas.paste(_c2, (114, 3), _c2)
except Exception:
    pass
layout["7.47"] = [114, 3, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 61, 60)
    canvas.paste(_c3, (310, 4), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 4, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 48, 70)
    canvas.paste(_c4, (1155, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1155, 0, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 59)
    canvas.paste(_c5, (249, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 5, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/06_icon_7.47.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (12, 72), _c6)
except Exception:
    pass
layout["7.47"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 95, 70)
    canvas.paste(_c7, (1211, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1211, 0, 1306, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 66)
    canvas.paste(_c8, (1325, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/09_icon_7.47.png
try:
    _c9 = get_crop(9, 90, 62)
    canvas.paste(_c9, (17, 3), _c9)
except Exception:
    pass
layout["7.47"] = [17, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/10_icon_What_date.png
try:
    _c10 = get_crop(10, 318, 73)
    canvas.paste(_c10, (558, 111), _c10)
except Exception:
    pass
layout["What_date?"] = [558, 111, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 63)
    canvas.paste(_c11, (384, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [384, 3, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 580, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/13_text_End_Date.png
try:
    _c13 = get_crop(13, 580, 144)
    canvas.paste(_c13, (48, 313), _c13)
except Exception:
    pass
layout["End_Date"] = [48, 313, 628, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_09_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-11/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
