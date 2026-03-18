# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_09
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12.png
# step_index: 9/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the UI mockup
# Uses provided canvas (1440x2960) and draw (ImageDraw)

# Colors
bg_color = (247, 247, 247)        # very light off-white background
status_bar_color = (90, 90, 90)   # dark grey status bar
modal_white = (255, 255, 255)     # white modal sheet
sep_color = (230, 230, 230)       # light separator lines
card_bg = (250, 250, 250)         # subtle card background
soft_shadow = (220, 220, 220)     # subtle shadow color

# Fill overall canvas background
draw.rectangle((0, 0, 1440, 2960), fill=bg_color)

# Status bar area (top ~56px)
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill=status_bar_color)

# Modal / sheet with rounded top corners (starts under status bar)
modal_x0, modal_y0 = 20, 40
modal_x1, modal_y1 = 1420, 2920
modal_radius = 36
draw.rounded_rectangle((modal_x0, modal_y0, modal_x1, modal_y1), radius=modal_radius, fill=modal_white)

# Subtle shadow line under the modal top edge to separate it from status bar
draw.line((modal_x0 + 4, modal_y0 + modal_radius//2, modal_x1 - 4, modal_y0 + modal_radius//2), fill=soft_shadow, width=1)

# Header divider under the toolbar/title area
header_div_y = modal_y0 + 120
draw.line((modal_x0 + 24, header_div_y, modal_x1 - 24, header_div_y), fill=sep_color, width=1)

# Section separators (light lines inside the modal)
separators = [
    modal_y0 + 320,   # after quantity section
    modal_y0 + 900,   # after price-per-ticket header area
    modal_y0 + 1510,  # after price graph/toggle area
    modal_y0 + 1760,  # before options group
    modal_y0 + 1960,  # after options header
    modal_y1 - 220    # above bottom CTA area
]
for y in separators:
    draw.line((modal_x0 + 24, y, modal_x1 - 24, y), fill=sep_color, width=1)

# Draw subtle rounded "cards" / group backgrounds for main sections (no content)
# Quantity group background (slight card)
q_x0, q_y0 = modal_x0 + 12, modal_y0 + 80
q_x1, q_y1 = modal_x1 - 12, modal_y0 + 320 - 8
draw.rounded_rectangle((q_x0, q_y0, q_x1, q_y1), radius=12, fill=card_bg, outline=None)

# Price per ticket group (white still but give a faint inset background block)
price_x0, price_y0 = modal_x0 + 12, modal_y0 + 340
price_x1, price_y1 = modal_x1 - 12, modal_y0 + 1510 - 8
draw.rectangle((price_x0, price_y0, price_x1, price_y1), fill=modal_white)
# Add a faint inner top border to separate the small title area from the graph area
draw.line((price_x0 + 12, price_y0 + 72, price_x1 - 12, price_y0 + 72), fill=sep_color, width=1)

# Graph/content area background (placeholder pale region to indicate content zone)
graph_x0 = price_x0 + 20
graph_y0 = price_y0 + 120
graph_x1 = price_x1 - 20
graph_y1 = price_y0 + 520
draw.rectangle((graph_x0, graph_y0, graph_x1, graph_y1), fill=card_bg, outline=None)

# Toggle area placeholder background row (right-aligned toggle sits visually on this row)
toggle_row_y = price_y0 + 560
draw.line((price_x0 + 12, toggle_row_y, price_x1 - 12, toggle_row_y), fill=sep_color, width=1)

# Options group card (subtle rounded block)
opts_x0, opts_y0 = modal_x0 + 12, modal_y0 + 1780
opts_x1, opts_y1 = modal_x1 - 12, modal_y0 + 2120
draw.rounded_rectangle((opts_x0, opts_y0, opts_x1, opts_y1), radius=12, fill=card_bg, outline=None)
# Small separator inside options area
draw.line((opts_x0 + 12, opts_y0 + 84, opts_x1 - 12, opts_y0 + 84), fill=sep_color, width=1)

# Large empty content area (below options) - very light background to indicate scannable area
content_x0, content_y0 = modal_x0 + 12, opts_y1 + 24
content_x1, content_y1 = modal_x1 - 12, modal_y1 - 260
draw.rectangle((content_x0, content_y0, content_x1, content_y1), fill=modal_white)

# Bottom fade/shadow bar above bottom controls (to give depth; don't draw any buttons)
fade_top = modal_y1 - 260
for i in range(20):
    alpha = int(220 - i * 8)
    if alpha < 0: alpha = 0
    shade = (240 - i, 240 - i, 240 - i)
    draw.line((modal_x0 + 12, fade_top + i, modal_x1 - 12, fade_top + i), fill=shade, width=1)

# Subtle vertical dividers for structure (do not create UI elements)
left_margin_x = modal_x0 + 36
right_margin_x = modal_x1 - 36
draw.line((left_margin_x, modal_y0 + 40, left_margin_x, modal_y1 - 40), fill=(245,245,245), width=1)
draw.line((right_margin_x, modal_y0 + 40, right_margin_x, modal_y1 - 40), fill=(245,245,245), width=1)

# Top-center small handle indicator (thin rounded rectangle) to hint sheet drag - very subtle
handle_w, handle_h = 140, 6
handle_x0 = (modal_x0 + modal_x1)//2 - handle_w//2
handle_y0 = modal_y0 + 12
draw.rounded_rectangle((handle_x0, handle_y0, handle_x0 + handle_w, handle_y0 + handle_h), radius=3, fill=(235,235,235))

# End of structure drawing - all shapes are purely background/structure (no icons/text/buttons)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/00_icon_View_1_000_listings.png
try:
    _c0 = get_crop(0, 554, 144)
    canvas.paste(_c0, (826, 2768), _c0)
except Exception:
    pass
layout["View_1,000+_listings"] = [826, 2768, 1380, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/01_icon_Any.png
try:
    _c1 = get_crop(1, 176, 110)
    canvas.paste(_c1, (60, 512), _c1)
except Exception:
    pass
layout["Any"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/02_icon_5.png
try:
    _c2 = get_crop(2, 144, 110)
    canvas.paste(_c2, (899, 512), _c2)
except Exception:
    pass
layout["5"] = [899, 512, 1043, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/03_icon_6.png
try:
    _c3 = get_crop(3, 144, 110)
    canvas.paste(_c3, (1062, 512), _c3)
except Exception:
    pass
layout["6"] = [1062, 512, 1206, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/04_icon_4.png
try:
    _c4 = get_crop(4, 144, 110)
    canvas.paste(_c4, (736, 512), _c4)
except Exception:
    pass
layout["4"] = [736, 512, 880, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/05_icon_7.png
try:
    _c5 = get_crop(5, 144, 110)
    canvas.paste(_c5, (1223, 512), _c5)
except Exception:
    pass
layout["7"] = [1223, 512, 1367, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/06_icon_3.png
try:
    _c6 = get_crop(6, 144, 110)
    canvas.paste(_c6, (573, 512), _c6)
except Exception:
    pass
layout["3"] = [573, 512, 717, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/07_icon_Any.png
try:
    _c7 = get_crop(7, 144, 110)
    canvas.paste(_c7, (412, 512), _c7)
except Exception:
    pass
layout["Any"] = [412, 512, 556, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/08_icon_Any.png
try:
    _c8 = get_crop(8, 144, 110)
    canvas.paste(_c8, (257, 512), _c8)
except Exception:
    pass
layout["Any"] = [257, 512, 401, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 101, 102)
    canvas.paste(_c9, (1277, 1346), _c9)
except Exception:
    pass
layout["icon_9"] = [1277, 1346, 1378, 1448]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 63, 58)
    canvas.paste(_c10, (243, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [243, 4, 306, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/11_icon_5.00_my.png
try:
    _c11 = get_crop(11, 51, 58)
    canvas.paste(_c11, (184, 3), _c11)
except Exception:
    pass
layout["5.00_my"] = [184, 3, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/12_icon_5.00_my.png
try:
    _c12 = get_crop(12, 59, 60)
    canvas.paste(_c12, (112, 2), _c12)
except Exception:
    pass
layout["5.00_my"] = [112, 2, 171, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 45, 60)
    canvas.paste(_c13, (1155, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [1155, 5, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 98, 57)
    canvas.paste(_c14, (1215, 5), _c14)
except Exception:
    pass
layout["icon_14"] = [1215, 5, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 55, 119)
    canvas.paste(_c15, (1385, 509), _c15)
except Exception:
    pass
layout["icon_15"] = [1385, 509, 1440, 628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 49, 53)
    canvas.paste(_c16, (1321, 6), _c16)
except Exception:
    pass
layout["icon_16"] = [1321, 6, 1370, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 56)
    canvas.paste(_c17, (317, 6), _c17)
except Exception:
    pass
layout["icon_17"] = [317, 6, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 49, 53)
    canvas.paste(_c18, (383, 8), _c18)
except Exception:
    pass
layout["icon_18"] = [383, 8, 432, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 102, 102)
    canvas.paste(_c19, (56, 1346), _c19)
except Exception:
    pass
layout["icon_19"] = [56, 1346, 158, 1448]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/20_icon_Price.png
try:
    _c20 = get_crop(20, 1440, 144)
    canvas.paste(_c20, (0, 1878), _c20)
except Exception:
    pass
layout["Price"] = [0, 1878, 1440, 2022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/21_icon_Price.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1251, 1500), _c21)
except Exception:
    pass
layout["Price"] = [1251, 1500, 1395, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/22_text_Filters.png
try:
    _c22 = get_crop(22, 1344, 156)
    canvas.paste(_c22, (48, 120), _c22)
except Exception:
    pass
layout["Filters"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/23_text_Quantity.png
try:
    _c23 = get_crop(23, 176, 110)
    canvas.paste(_c23, (60, 512), _c23)
except Exception:
    pass
layout["Quantity"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/24_text_Price_per_ticket.png
try:
    _c24 = get_crop(24, 176, 110)
    canvas.paste(_c24, (60, 512), _c24)
except Exception:
    pass
layout["Price_per_ticket"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/25_text_S174-S11_944.png
try:
    _c25 = get_crop(25, 1440, 139)
    canvas.paste(_c25, (0, 910), _c25)
except Exception:
    pass
layout["S174-S11,944"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/26_text_price_based_on_filters_is_S577.png
try:
    _c26 = get_crop(26, 1440, 139)
    canvas.paste(_c26, (0, 910), _c26)
except Exception:
    pass
layout["price_based_on_filters_is"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/27_text_Show_prices_with_fees.png
try:
    _c27 = get_crop(27, 1440, 144)
    canvas.paste(_c27, (0, 1500), _c27)
except Exception:
    pass
layout["Show_prices_with_fees"] = [0, 1500, 1440, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/28_text_Options.png
try:
    _c28 = get_crop(28, 192, 61)
    canvas.paste(_c28, (55, 1784), _c28)
except Exception:
    pass
layout["Options"] = [55, 1784, 247, 1845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/29_text_Sort_by.png
try:
    _c29 = get_crop(29, 178, 63)
    canvas.paste(_c29, (55, 1923), _c29)
except Exception:
    pass
layout["Sort_by"] = [55, 1923, 233, 1986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_09_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-12/30_text_Clear_all.png
try:
    _c30 = get_crop(30, 193, 144)
    canvas.paste(_c30, (60, 2766), _c30)
except Exception:
    pass
layout["Clear_all"] = [60, 2766, 253, 2910]
