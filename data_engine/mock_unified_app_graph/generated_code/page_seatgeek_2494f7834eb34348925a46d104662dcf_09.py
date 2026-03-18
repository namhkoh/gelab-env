# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_09
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12.png
# step_index: 9/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for the mobile page
# Uses provided variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_* (unused)

# Colors
bg = "#eef1f3"            # overall page background (very light gray)
status_bg = "#e6e7e9"     # status bar background (slightly darker)
header_bg = "#ffffff"     # header/toolbar background (white pill)
header_border = "#e3e5e7" # subtle border for header
card_bg = "#ffffff"       # card background (white)
muted_panel = "#f6f7f8"   # subtle panel behind map
muted_border = "#dde0e3"  # borders for panels
shadow = "#d9dbdd"        # faux shadow color
divider = "#e6e7e8"       # thin separators

W, H = canvas.size

# 1) Page background
draw.rectangle((0, 0, W, H), fill=bg)

# 2) Status bar area at the top (~0..84 px)
status_h = 84
draw.rectangle((0, 0, W, status_h), fill=status_bg)

# 3) Header / toolbar pill (rounded) that sits below the status bar
header_top = status_h + 8
header_bottom = header_top + 112
header_left = 48
header_right = W - 48
draw.rounded_rectangle((header_left, header_top, header_right, header_bottom),
                       radius=56, fill=header_bg, outline=header_border, width=1)

# subtle shadow below header
draw.rectangle((header_left+4, header_bottom+2, header_right-4, header_bottom+6), fill=shadow)

# 4) Filter row background band (subtle, behind the filter pills)
# Keep it visually distinct but soft so it doesn't conflict with pasted pill elements
filter_band_top = header_bottom + 28
filter_band_bottom = filter_band_top + 140
band_margin = 28
draw.rounded_rectangle((band_margin, filter_band_top, W - band_margin, filter_band_bottom),
                       radius=24, fill=bg, outline=muted_border, width=1)

# 5) Large seating/map background panel (centered)
# This is the content area that holds the seating map; draw a pale panel with border
map_panel_top = filter_band_bottom + 36
map_panel_bottom = map_panel_top + 1080
map_panel_left = 240
map_panel_right = W - 240
draw.rounded_rectangle((map_panel_left, map_panel_top, map_panel_right, map_panel_bottom),
                       radius=12, fill=muted_panel, outline=muted_border, width=2)

# soft inner highlight (to mimic inset)
inner_pad = 12
draw.rounded_rectangle((map_panel_left + inner_pad, map_panel_top + inner_pad,
                        map_panel_right - inner_pad, map_panel_bottom - inner_pad),
                       radius=10, outline="#f0f2f3", width=1)

# 6) Separator line between the map area and the lower card section
sep_y = map_panel_bottom + 42
draw.line((48, sep_y, W - 48, sep_y), fill=divider, width=1)

# 7) Bottom card panel for "Box office & resale"
card_top = sep_y + 24
card_left = 0
card_right = W
card_bottom = H
card_radius = 36
# Draw a shadow band above the card
draw.rectangle((card_left, card_top - 10, card_right, card_top), fill=shadow)
# Card itself (rounded top corners)
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom),
                       radius=card_radius, fill=card_bg, outline=muted_border, width=1)

# 8) Card header divider (thin line under the card header area)
card_header_h = 120
card_header_y = card_top + card_header_h
draw.line((48, card_header_y, W - 48, card_header_y), fill=divider, width=1)

# 9) Small subsection backgrounds inside the card (e.g., thumbnail background)
# These are structural backgrounds only; content (images/text) will be pasted on top.
thumb_x = 64
thumb_y = card_header_y + 80
thumb_w = 220
thumb_h = 160
draw.rounded_rectangle((thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h),
                       radius=12, fill="#fff5ef", outline=muted_border, width=1)

# 10) Light chip background behind "Limited View" area (structural only)
chip_w = 220
chip_h = 56
chip_x = 64
chip_y = thumb_y + thumb_h + 56
draw.rounded_rectangle((chip_x, chip_y, chip_x + chip_w, chip_y + chip_h),
                       radius=12, fill="#fff4ee", outline="#f0d6c6", width=1)

# 11) Vertical spacing helpers (thin separators) to divide content areas subtly
draw.line((48, thumb_y - 24, 48, card_bottom - 48), fill="#fbfbfb", width=1)
draw.line((W - 48, thumb_y - 24, W - 48, card_bottom - 48), fill="#fbfbfb", width=1)

# End structural drawing - actual icons/text/images will be pasted on top by the pipeline.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (542, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [542, 312, 877, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/01_icon_2_tickets.png
try:
    _c1 = get_crop(1, 266, 108)
    canvas.paste(_c1, (240, 312), _c1)
except Exception:
    pass
layout["2_tickets"] = [240, 312, 506, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/02_icon_Hide_resale.png
try:
    _c2 = get_crop(2, 315, 108)
    canvas.paste(_c2, (913, 312), _c2)
except Exception:
    pass
layout["Hide_resale"] = [913, 312, 1228, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/03_icon_9.4.png
try:
    _c3 = get_crop(3, 1440, 588)
    canvas.paste(_c3, (0, 2355), _c3)
except Exception:
    pass
layout["9.4"] = [0, 2355, 1440, 2943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/05_icon_Acces.png
try:
    _c5 = get_crop(5, 176, 108)
    canvas.paste(_c5, (1264, 312), _c5)
except Exception:
    pass
layout["Acces="] = [1264, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 65)
    canvas.paste(_c6, (1152, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1152, 1, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/07_icon_6.52_Wy.png
try:
    _c7 = get_crop(7, 66, 63)
    canvas.paste(_c7, (111, 1), _c7)
except Exception:
    pass
layout["6.52_Wy"] = [111, 1, 177, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 63, 59)
    canvas.paste(_c8, (243, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [243, 3, 306, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/09_icon_6.52_Wy.png
try:
    _c9 = get_crop(9, 55, 59)
    canvas.paste(_c9, (182, 2), _c9)
except Exception:
    pass
layout["6.52_Wy"] = [182, 2, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/10_icon_0.png
try:
    _c10 = get_crop(10, 102, 63)
    canvas.paste(_c10, (1213, 1), _c10)
except Exception:
    pass
layout["0#"] = [1213, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/11_icon_0.png
try:
    _c11 = get_crop(11, 156, 156)
    canvas.paste(_c11, (1236, 120), _c11)
except Exception:
    pass
layout["0#"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 58)
    canvas.paste(_c12, (1320, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1320, 3, 1372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 59, 60)
    canvas.paste(_c13, (314, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [314, 3, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/14_icon_Include_fees.png
try:
    _c14 = get_crop(14, 1344, 156)
    canvas.paste(_c14, (48, 120), _c14)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/15_icon_The_Book_of_Mormon.png
try:
    _c15 = get_crop(15, 51, 62)
    canvas.paste(_c15, (382, 1), _c15)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [382, 1, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/16_icon_Sort_by_price.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 1989), _c16)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/17_icon_Limited_View.png
try:
    _c17 = get_crop(17, 1440, 588)
    canvas.paste(_c17, (0, 2355), _c17)
except Exception:
    pass
layout["Limited_View"] = [0, 2355, 1440, 2943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/18_icon_Oy_ANane_Atthebox_Orrce.png
try:
    _c18 = get_crop(18, 335, 108)
    canvas.paste(_c18, (542, 312), _c18)
except Exception:
    pass
layout["Oy_ANane_Atthebox_Orrce"] = [542, 312, 877, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/19_icon_Amazing_deal.png
try:
    _c19 = get_crop(19, 1440, 588)
    canvas.paste(_c19, (0, 2355), _c19)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2355, 1440, 2943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/20_icon_6.52_Wy.png
try:
    _c20 = get_crop(20, 99, 65)
    canvas.paste(_c20, (5, 0), _c20)
except Exception:
    pass
layout["6.52_Wy"] = [5, 0, 104, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/21_text_Box_office_resale.png
try:
    _c21 = get_crop(21, 489, 54)
    canvas.paste(_c21, (58, 2033), _c21)
except Exception:
    pass
layout["Box_office_&_resale"] = [58, 2033, 547, 2087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/22_text_We_sell_box_office_and_resale_tickets._R.png
try:
    _c22 = get_crop(22, 1440, 588)
    canvas.paste(_c22, (0, 2355), _c22)
except Exception:
    pass
layout["We_sell_box_office_and_€_"] = [0, 2355, 1440, 2943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/23_text_below_face_value.png
try:
    _c23 = get_crop(23, 350, 54)
    canvas.paste(_c23, (56, 2250), _c23)
except Exception:
    pass
layout["below_face_value"] = [56, 2250, 406, 2304]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/24_clickable_Back.png
try:
    _c24 = get_crop(24, 156, 156)
    canvas.paste(_c24, (48, 120), _c24)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_09_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-12/25_clickable_The_Book_of_Mormon.png
try:
    _c25 = get_crop(25, 413, 156)
    canvas.paste(_c25, (204, 120), _c25)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [204, 120, 617, 276]
