# page_id: page_seatgeek_1e6c9e893d9e4bc99959744188677162_08
# screenshot: 2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11.png
# step_index: 8/8
# task: Open SeatGeek. Search "Radio City Music Hall" and then add the venue to favorite. Who are the performers of the top recommended event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and layout drawing for SeatGeek-like event page
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_white = (255, 255, 255)
status_bar_color = (245, 245, 245)
status_bar_border = (225, 225, 225)
blue_top_a = (12, 142, 237)
blue_top_b = (57, 179, 243)
card_shadow = (232, 232, 232)
card_white = (255, 255, 255)
sep_color = (235, 235, 235)
thin_sep = (245, 245, 245)

# Fill overall background (dominant color)
draw.rectangle([(0, 0), (w, h)], fill=bg_white)

# Status bar area (top ~50px)
status_h = 50
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)
draw.line([(0, status_h - 1), (w, status_h - 1)], fill=status_bar_border, width=1)

# Hero/banner area with subtle vertical gradient (from y=status_h to ~520)
hero_top = status_h
hero_bottom = 520
for yy in range(hero_top, hero_bottom):
    t = (yy - hero_top) / max(1, hero_bottom - hero_top)
    r = int(blue_top_a[0] + (blue_top_b[0] - blue_top_a[0]) * t)
    g = int(blue_top_a[1] + (blue_top_b[1] - blue_top_a[1]) * t)
    b = int(blue_top_a[2] + (blue_top_b[2] - blue_top_a[2]) * t)
    draw.line([(0, yy), (w, yy)], fill=(r, g, b))

# Slanted white overlay across bottom of hero to emulate diagonal card cut-out
# Points chosen to match a left-lower to right-higher diagonal slope
slant = [(0, 460), (w, 360), (w, hero_bottom + 80), (0, hero_bottom + 80)]
draw.polygon(slant, fill=card_white)

# Main header/card area behind title and action buttons
header_top = 420
header_bottom = 1120
header_margin = 20
# subtle shadow behind the header card
draw.rounded_rectangle(
    [(header_margin - 2, header_top + 8), (w - header_margin + 2, header_bottom + 10)],
    radius=16,
    fill=card_shadow
)
# actual card (white) - this is the background only (no text/icons)
draw.rounded_rectangle(
    [(header_margin, header_top), (w - header_margin, header_bottom)],
    radius=16,
    fill=card_white
)

# Thin divider below header card
draw.line([(header_margin + 8, header_bottom), (w - header_margin - 8, header_bottom)], fill=sep_color, width=1)

# Location/venue section background (keeps same white canvas but add subtle top/bottom separation)
loc_top = header_bottom + 10
loc_bottom = 1700
# light separator above location group
draw.line([(16, loc_top), (w - 16, loc_top)], fill=thin_sep, width=1)
# subtle background band to separate sections (very light)
draw.rectangle([(0, loc_top), (w, loc_top + 8)], fill=thin_sep)

# Divider between location and "more events" area
divider_1 = 1640
draw.line([(0, divider_1), (w, divider_1)], fill=sep_color, width=1)

# Performers section card with shadow and rounded corners
perf_top = 1720
perf_bottom = 2360
perf_margin = 20
# shadow
draw.rounded_rectangle(
    [(perf_margin - 2, perf_top + 8), (w - perf_margin + 2, perf_bottom + 10)],
    radius=14,
    fill=card_shadow
)
# performers card background
draw.rounded_rectangle(
    [(perf_margin, perf_top), (w - perf_margin, perf_bottom)],
    radius=14,
    fill=card_white
)

# Subtle separator lines inside performers card (top and bottom)
draw.line([(perf_margin + 12, perf_top), (w - perf_margin - 12, perf_top)], fill=thin_sep, width=1)
draw.line([(perf_margin + 12, perf_bottom), (w - perf_margin - 12, perf_bottom)], fill=sep_color, width=1)

# Box office area and separators
box_top = perf_bottom + 20
box_bottom = 2580
draw.rectangle([(0, box_top), (w, box_bottom)], fill=card_white)
draw.line([(16, box_top), (w - 16, box_top)], fill=sep_color, width=1)
draw.line([(16, box_bottom), (w - 16, box_bottom)], fill=sep_color, width=1)

# Bottom "View tickets" area (kept as background only, text/buttons will be pasted separately)
view_tickets_top = 2581
view_tickets_bottom = view_tickets_top + 144
# very slight grey background to separate from above content
draw.rectangle([(0, view_tickets_top), (w, view_tickets_bottom)], fill=card_white)
draw.line([(0, view_tickets_top), (w, view_tickets_top)], fill=sep_color, width=1)

# Final very light vertical guide lines/margins (do not draw any icons/text)
# These help define the content area margins only
left_g = 40
right_g = w - 40
draw.line([(left_g, view_tickets_bottom + 4), (left_g, h - 1)], fill=thin_sep, width=1)
draw.line([(right_g, view_tickets_bottom + 4), (right_g, h - 1)], fill=thin_sep, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/00_icon_Share.png
try:
    _c0 = get_crop(0, 312, 153)
    canvas.paste(_c0, (552, 978), _c0)
except Exception:
    pass
layout["Share"] = [552, 978, 864, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/01_icon_Track_event.png
try:
    _c1 = get_crop(1, 444, 153)
    canvas.paste(_c1, (60, 978), _c1)
except Exception:
    pass
layout["Track_event"] = [60, 978, 504, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/02_icon_Laufey_with_Wasia_Project.png
try:
    _c2 = get_crop(2, 444, 153)
    canvas.paste(_c2, (60, 978), _c2)
except Exception:
    pass
layout["Laufey_with_Wasia_Project"] = [60, 978, 504, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/03_icon_8.33_my.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (24, 84), _c3)
except Exception:
    pass
layout["8.33_my"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/04_icon_Performers.png
try:
    _c4 = get_crop(4, 1416, 179)
    canvas.paste(_c4, (12, 1992), _c4)
except Exception:
    pass
layout["Performers"] = [12, 1992, 1428, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 61, 64)
    canvas.paste(_c5, (241, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [241, 3, 302, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 47, 70)
    canvas.paste(_c6, (1155, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1155, 0, 1202, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/07_icon_8.33_my.png
try:
    _c7 = get_crop(7, 51, 65)
    canvas.paste(_c7, (183, 2), _c7)
except Exception:
    pass
layout["8.33_my"] = [183, 2, 234, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/08_icon_8.33_my.png
try:
    _c8 = get_crop(8, 57, 67)
    canvas.paste(_c8, (115, 0), _c8)
except Exception:
    pass
layout["8.33_my"] = [115, 0, 172, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/09_icon_14_events.png
try:
    _c9 = get_crop(9, 1416, 179)
    canvas.paste(_c9, (12, 2171), _c9)
except Exception:
    pass
layout["14_events"] = [12, 2171, 1428, 2350]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 68)
    canvas.paste(_c10, (1320, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1320, 1, 1370, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 55, 63)
    canvas.paste(_c11, (313, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [313, 4, 368, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/12_icon_View_tickets.png
try:
    _c12 = get_crop(12, 1440, 144)
    canvas.paste(_c12, (0, 2581), _c12)
except Exception:
    pass
layout["View_tickets"] = [0, 2581, 1440, 2725]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 59, 70)
    canvas.paste(_c13, (1213, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1213, 0, 1272, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 47, 67)
    canvas.paste(_c14, (1270, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1270, 2, 1317, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/15_icon_Wasia_Project.png
try:
    _c15 = get_crop(15, 1416, 179)
    canvas.paste(_c15, (12, 1992), _c15)
except Exception:
    pass
layout["Wasia_Project"] = [12, 1992, 1428, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 46, 64)
    canvas.paste(_c16, (383, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [383, 1, 429, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/17_icon_Music_Hall.png
try:
    _c17 = get_crop(17, 1440, 113)
    canvas.paste(_c17, (0, 1553), _c17)
except Exception:
    pass
layout["Music_Hall"] = [0, 1553, 1440, 1666]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/18_text_Location.png
try:
    _c18 = get_crop(18, 209, 52)
    canvas.paste(_c18, (56, 1263), _c18)
except Exception:
    pass
layout["Location"] = [56, 1263, 265, 1315]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/19_text_Get_directions.png
try:
    _c19 = get_crop(19, 1440, 113)
    canvas.paste(_c19, (0, 1553), _c19)
except Exception:
    pass
layout["Get_directions"] = [0, 1553, 1440, 1666]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/20_text_More_events_at_Radio.png
try:
    _c20 = get_crop(20, 1440, 113)
    canvas.paste(_c20, (0, 1666), _c20)
except Exception:
    pass
layout["More_events_at_Radio"] = [0, 1666, 1440, 1779]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/21_text_Music_Hall.png
try:
    _c21 = get_crop(21, 1440, 113)
    canvas.paste(_c21, (0, 1666), _c21)
except Exception:
    pass
layout["Music_Hall"] = [0, 1666, 1440, 1779]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/22_text_Performers.png
try:
    _c22 = get_crop(22, 256, 54)
    canvas.paste(_c22, (55, 1893), _c22)
except Exception:
    pass
layout["Performers"] = [55, 1893, 311, 1947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/23_text_Wasia_Project.png
try:
    _c23 = get_crop(23, 304, 57)
    canvas.paste(_c23, (250, 2206), _c23)
except Exception:
    pass
layout["Wasia_Project"] = [250, 2206, 554, 2263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/24_text_14_events.png
try:
    _c24 = get_crop(24, 191, 41)
    canvas.paste(_c24, (249, 2274), _c24)
except Exception:
    pass
layout["14_events"] = [249, 2274, 440, 2315]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_08_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-11/25_text_Box_office.png
try:
    _c25 = get_crop(25, 232, 49)
    canvas.paste(_c25, (56, 2484), _c25)
except Exception:
    pass
layout["Box_office"] = [56, 2484, 288, 2533]
