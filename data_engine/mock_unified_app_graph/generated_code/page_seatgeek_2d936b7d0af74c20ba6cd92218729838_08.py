# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_08
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11.png
# step_index: 8/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
bg_color = (250, 250, 250)        # very light off-white background
status_bar_color = (235, 235, 235) # slight gray for status bar
search_bg = (245, 245, 245)       # search field background
content_card = (255, 255, 255)    # white card for main content
divider_color = (225, 225, 225)   # subtle divider lines
bottom_bar_color = (255, 255, 255)
shadow_color = (240, 240, 240)

# Full canvas fill
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar (top area)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill=status_bar_color)

# Main content card (large rounded panel behind list and suggestions)
card_margin_x = 32
card_top = 88
card_bottom = 2680
card_radius = 28
draw.rounded_rectangle(
    [(card_margin_x, card_top), (1440 - card_margin_x, card_bottom)],
    radius=card_radius,
    fill=content_card,
    outline=None
)

# Search bar background (rounded)
search_left = 48
search_right = 1440 - 48
search_top = 96
search_bottom = search_top + 144
search_radius = 16
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=search_bg,
    outline=(230, 230, 230),
    width=1
)

# subtle shadow under search bar
shadow_top = search_bottom
for i, alpha_offset in enumerate(range(4)):
    y = shadow_top + i
    shade = int(245 - i * 3)
    draw.line([(search_left+2, y), (search_right-2, y)], fill=(shade, shade, shade))

# Primary horizontal separators for major sections
# Separator under recent searches list (approx)
sep_positions = [
    360,  # after search area / heading
    1312, # after recent searches block
    1496, # before suggestions list
    2040  # lower area separator (if needed)
]
for y in sep_positions:
    draw.line([(card_margin_x + 8, y), (1440 - card_margin_x - 8, y)], fill=divider_color, width=1)

# Sub-item separators for the recent searches list
# Use approximate item heights (168) starting around the first item area
item_start_y = 471
item_height = 168
for i in range(1, 5):
    y = item_start_y + i * item_height
    draw.line([(card_margin_x + 8, y), (1440 - card_margin_x - 8, y)], fill=divider_color, width=1)

# Sub-item separators for Suggestions block (three suggestion rows)
suggest_start = 1520
for i in range(1, 3):
    y = suggest_start + i * 168
    draw.line([(card_margin_x + 8, y), (1440 - card_margin_x - 8, y)], fill=divider_color, width=1)

# Bottom navigation bar background and top border
bottom_top = 2720
draw.rectangle([(0, bottom_top), (1440, 2960)], fill=bottom_bar_color)
draw.line([(0, bottom_top), (1440, bottom_top)], fill=divider_color, width=1)

# Very subtle vignette/shadow near card edges to lift the card slightly
# top shadow
for i in range(6):
    alpha = 1 - (i / 12.0)
    shade = int(235 + i)
    y = card_top + i
    draw.line([(card_margin_x+2, y), (1440 - card_margin_x-2, y)], fill=(shade, shade, shade))

# right-side subtle inner border to define content area (very light)
draw.line([(1440 - card_margin_x, card_top + 6), (1440 - card_margin_x, card_bottom - 6)], fill=(245,245,245), width=1)
# left-side subtle inner border
draw.line([(card_margin_x, card_top + 6), (card_margin_x, card_bottom - 6)], fill=(245,245,245), width=1)

# End of background and structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/00_icon_Morm.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Morm"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/01_icon_Suggestions.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 1143), _c1)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 47, 70)
    canvas.paste(_c2, (1153, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/03_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 975), _c3)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/04_icon_The_Lion_King.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 975), _c4)
except Exception:
    pass
layout["The_Lion_King"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/05_icon_Tracking.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (864, 2792), _c5)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/06_icon_The_Book_of_Mormon.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 639), _c6)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 66, 63)
    canvas.paste(_c7, (242, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [242, 2, 308, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/08_icon_Tickets.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (576, 2792), _c8)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/09_icon_Just_Announced_by_My_Performers.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 1688), _c9)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 96, 69)
    canvas.paste(_c10, (1217, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1217, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/11_icon_Browse.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (0, 2792), _c11)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/12_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 807), _c12)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/13_icon_Los_Angeles_Clippers.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 471), _c13)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/15_icon_Events_by_My_Performers.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 1520), _c15)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/16_icon_The_Phantom_of_the_Opera.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1143), _c16)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/17_icon_6.54_Wy.png
try:
    _c17 = get_crop(17, 47, 63)
    canvas.paste(_c17, (186, 1), _c17)
except Exception:
    pass
layout["6.54_Wy"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/18_icon_6.54_Wy.png
try:
    _c18 = get_crop(18, 168, 144)
    canvas.paste(_c18, (48, 120), _c18)
except Exception:
    pass
layout["6.54_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/19_icon_6.54_Wy.png
try:
    _c19 = get_crop(19, 56, 64)
    canvas.paste(_c19, (114, 0), _c19)
except Exception:
    pass
layout["6.54_Wy"] = [114, 0, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 52, 69)
    canvas.paste(_c21, (1319, 0), _c21)
except Exception:
    pass
layout["icon_21"] = [1319, 0, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/22_icon_Morm.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 639), _c22)
except Exception:
    pass
layout["Morm"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/23_icon_Morm.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 807), _c23)
except Exception:
    pass
layout["Morm"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/25_icon_Just_Announced_by_My_Performers.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1856), _c25)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/26_text_Performer_event_or_venue.png
try:
    _c26 = get_crop(26, 1032, 144)
    canvas.paste(_c26, (216, 120), _c26)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_08_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-11/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
