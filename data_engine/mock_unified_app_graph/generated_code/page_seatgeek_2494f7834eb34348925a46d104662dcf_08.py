# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_08
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11.png
# step_index: 8/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile page.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_light = (241, 243, 245)        # dominant light grey background
status_bar_bg = (233, 235, 237)  # slightly darker for status bar area
search_bg = (255, 255, 255)      # white search pill
search_outline = (220, 223, 226)
pill_inactive = (250, 251, 252)
pill_outline = (224, 226, 228)
pill_active = (34, 34, 34)       # dark active pill (no text drawn)
modal_shadow = (226, 229, 231)
modal_bg = (255, 255, 255)
modal_outline = (236, 238, 239)
divider = (232, 234, 235)
handle_color = (210, 213, 216)

# Fill full background
draw.rectangle((0, 0, w, h), fill=bg_light)

# Status bar area at top (~80px)
status_h = 80
draw.rectangle((0, 0, w, status_h), fill=status_bar_bg)

# Search/header pill (rounded) - behind nav icons/text (which will be pasted later)
search_margin_x = 60
search_top = 90
search_bottom = 220
search_radius = 60
draw.rounded_rectangle(
    (search_margin_x, search_top, w - search_margin_x, search_bottom),
    radius=search_radius,
    fill=search_bg,
    outline=search_outline,
    width=2
)

# Row of filter pills below the header - draw only pill backgrounds (no text/icons)
pill_y1 = 300
pill_height = 130
pill_y2 = pill_y1 + pill_height

# pill positions chosen to align with detected element sizes (background only)
pill_rects = [
    (60, pill_y1, 339, pill_y2),     # Quantity pill background
    (349, pill_y1, 693, pill_y2),    # Include fees (active)
    (703, pill_y1, 1025, pill_y2),   # Hide resale
    (1035, pill_y1, 1216, pill_y2)   # Access / other pill
]

# Draw inactive pills first
for i, rect in enumerate(pill_rects):
    x1, y1, x2, y2 = rect
    radius = int(pill_height / 2)
    if i == 1:
        # active pill dark background (no text)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=pill_active)
    else:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=pill_inactive, outline=pill_outline, width=2)

# Modal sheet: large white rounded rectangle with subtle shadow (sheet that contains the list)
modal_left = 40
modal_right = w - 40
# Place modal top below pills with comfortable gap
modal_top = pill_y2 - 40
modal_bottom = h - 20
modal_radius = 40

# Draw a shadow behind the modal by drawing a slightly larger rounded rect in shadow color
shadow_offset = 10
draw.rounded_rectangle(
    (modal_left + 2, modal_top + shadow_offset, modal_right - 2, modal_bottom + shadow_offset),
    radius=modal_radius + 4,
    fill=modal_shadow
)

# Modal white background and thin outline
draw.rounded_rectangle(
    (modal_left, modal_top, modal_right, modal_bottom),
    radius=modal_radius,
    fill=modal_bg,
    outline=modal_outline,
    width=2
)

# Modal top subtle handle (small rounded capsule) - indicates draggable sheet
handle_w = 160
handle_h = 12
handle_x1 = (w - handle_w) / 2
handle_x2 = handle_x1 + handle_w
handle_y1 = modal_top + 18
handle_y2 = handle_y1 + handle_h
draw.rounded_rectangle((handle_x1, handle_y1, handle_x2, handle_y2), radius=handle_h//2, fill=handle_color)

# Divider line under modal header area (thin)
divider_y = modal_top + 110
draw.line((modal_left + 10, divider_y, modal_right - 10, divider_y), fill=divider, width=2)

# Additional subtle separators to structure the modal content area (do not draw list item boxes)
# Draw faint horizontal separators spaced as visual guides (these will sit behind pasted items)
sep_x1 = modal_left + 20
sep_x2 = modal_right - 20
sep_start = divider_y + 40
sep_spacing = 170
for i in range(1, 8):
    y = sep_start + i * sep_spacing
    if y < modal_bottom - 60:
        draw.line((sep_x1, y, sep_x2, y), fill=(245, 246, 247), width=1)

# Top toolbar subtle bottom border (below the search pill area)
toolbar_bottom = search_bottom + 20
draw.line((20, toolbar_bottom, w - 20, toolbar_bottom), fill=(225, 227, 229), width=1)

# Small rounded background behind the app title area (left side behind nav/back icon)
# This is structural only - icons/text will be pasted on top.
title_bg_left = 40
title_bg_right = 520
title_bg_top = search_top + 12
title_bg_bottom = search_top + 92
draw.rounded_rectangle((title_bg_left, title_bg_top, title_bg_right, title_bg_bottom), radius=40, fill=(250,250,250))

# Right side small info pill background (structural)
info_pill_w = 120
info_pill_h = 120
info_x1 = w - 60 - info_pill_w
info_x2 = info_x1 + info_pill_w
info_y1 = search_top + 12
info_y2 = info_y1 + info_pill_h
draw.rounded_rectangle((info_x1, info_y1, info_x2, info_y2), radius=36, fill=(250,250,250), outline=(230,231,233), width=1)

# Final subtle vignette line at bottom of modal (to separate app bottom area)
bottom_div_y = modal_bottom - 80
draw.line((modal_left + 20, bottom_div_y, modal_right - 20, bottom_div_y), fill=(240,241,242), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/00_icon_tickets.png
try:
    _c0 = get_crop(0, 1320, 157)
    canvas.paste(_c0, (60, 1513), _c0)
except Exception:
    pass
layout["tickets"] = [60, 1513, 1380, 1670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/01_icon_Quantity.png
try:
    _c1 = get_crop(1, 1320, 157)
    canvas.paste(_c1, (60, 693), _c1)
except Exception:
    pass
layout["Quantity"] = [60, 693, 1380, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/02_icon_3_tickets.png
try:
    _c2 = get_crop(2, 1320, 157)
    canvas.paste(_c2, (60, 1308), _c2)
except Exception:
    pass
layout["3_tickets"] = [60, 1308, 1380, 1465]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/03_icon_5_tickets.png
try:
    _c3 = get_crop(3, 1320, 157)
    canvas.paste(_c3, (60, 1718), _c3)
except Exception:
    pass
layout["5_tickets"] = [60, 1718, 1380, 1875]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/04_icon_9_tickets.png
try:
    _c4 = get_crop(4, 1320, 157)
    canvas.paste(_c4, (60, 2538), _c4)
except Exception:
    pass
layout["9_tickets"] = [60, 2538, 1380, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/05_icon_8_tickets.png
try:
    _c5 = get_crop(5, 1320, 157)
    canvas.paste(_c5, (60, 2333), _c5)
except Exception:
    pass
layout["8_tickets"] = [60, 2333, 1380, 2490]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/06_icon_2_tickets.png
try:
    _c6 = get_crop(6, 1320, 157)
    canvas.paste(_c6, (60, 1103), _c6)
except Exception:
    pass
layout["2_tickets"] = [60, 1103, 1380, 1260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/07_icon_7_tickets.png
try:
    _c7 = get_crop(7, 1320, 157)
    canvas.paste(_c7, (60, 2128), _c7)
except Exception:
    pass
layout["7_tickets"] = [60, 2128, 1380, 2285]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/08_icon_6_tickets.png
try:
    _c8 = get_crop(8, 1320, 157)
    canvas.paste(_c8, (60, 1923), _c8)
except Exception:
    pass
layout["6_tickets"] = [60, 1923, 1380, 2080]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/09_icon_quantity.png
try:
    _c9 = get_crop(9, 1320, 157)
    canvas.paste(_c9, (60, 898), _c9)
except Exception:
    pass
layout["quantity"] = [60, 898, 1380, 1055]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/10_icon_Include_fees.png
try:
    _c10 = get_crop(10, 344, 126)
    canvas.paste(_c10, (535, 309), _c10)
except Exception:
    pass
layout["Include_fees"] = [535, 309, 879, 435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/11_icon_Hide_resale.png
try:
    _c11 = get_crop(11, 322, 126)
    canvas.paste(_c11, (909, 309), _c11)
except Exception:
    pass
layout["Hide_resale"] = [909, 309, 1231, 435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/12_icon_Quantity.png
try:
    _c12 = get_crop(12, 279, 130)
    canvas.paste(_c12, (231, 308), _c12)
except Exception:
    pass
layout["Quantity"] = [231, 308, 510, 438]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/13_icon_Acces.png
try:
    _c13 = get_crop(13, 181, 135)
    canvas.paste(_c13, (1259, 308), _c13)
except Exception:
    pass
layout["Acces="] = [1259, 308, 1440, 443]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/14_icon_10_tickets.png
try:
    _c14 = get_crop(14, 1320, 157)
    canvas.paste(_c14, (60, 2743), _c14)
except Exception:
    pass
layout["10+_tickets"] = [60, 2743, 1380, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/15_icon_Acces.png
try:
    _c15 = get_crop(15, 102, 109)
    canvas.paste(_c15, (1257, 146), _c15)
except Exception:
    pass
layout["Acces="] = [1257, 146, 1359, 255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 66)
    canvas.paste(_c16, (1152, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1152, 1, 1203, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/17_icon_my.png
try:
    _c17 = get_crop(17, 68, 67)
    canvas.paste(_c17, (241, 0), _c17)
except Exception:
    pass
layout["my"] = [241, 0, 309, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 105, 65)
    canvas.paste(_c18, (1211, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1211, 0, 1316, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/19_icon_my.png
try:
    _c19 = get_crop(19, 71, 66)
    canvas.paste(_c19, (108, 0), _c19)
except Exception:
    pass
layout["my"] = [108, 0, 179, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 54, 62)
    canvas.paste(_c20, (1319, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [1319, 1, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/21_icon_my.png
try:
    _c21 = get_crop(21, 56, 65)
    canvas.paste(_c21, (180, 1), _c21)
except Exception:
    pass
layout["my"] = [180, 1, 236, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/22_icon_Tit.png
try:
    _c22 = get_crop(22, 168, 137)
    canvas.paste(_c22, (39, 307), _c22)
except Exception:
    pass
layout["Tit"] = [39, 307, 207, 444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/23_icon_The_Book_of_Mormon.png
try:
    _c23 = get_crop(23, 61, 66)
    canvas.paste(_c23, (313, 1), _c23)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [313, 1, 374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/24_icon_The_Book_of_Mormon.png
try:
    _c24 = get_crop(24, 52, 66)
    canvas.paste(_c24, (382, 1), _c24)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [382, 1, 434, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/25_icon_Quantity.png
try:
    _c25 = get_crop(25, 1320, 157)
    canvas.paste(_c25, (60, 693), _c25)
except Exception:
    pass
layout["Quantity"] = [60, 693, 1380, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/26_icon_Include_fees.png
try:
    _c26 = get_crop(26, 1320, 157)
    canvas.paste(_c26, (60, 693), _c26)
except Exception:
    pass
layout["Include_fees"] = [60, 693, 1380, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_08_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-11/27_text_6.51.png
try:
    _c27 = get_crop(27, 87, 45)
    canvas.paste(_c27, (20, 15), _c27)
except Exception:
    pass
layout["6.51"] = [20, 15, 107, 60]
