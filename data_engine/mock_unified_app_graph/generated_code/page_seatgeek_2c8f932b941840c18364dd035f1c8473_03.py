# page_id: page_seatgeek_2c8f932b941840c18364dd035f1c8473_03
# screenshot: 2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6.png
# step_index: 3/8
# task: Open SeatGeek. Search "Beatles Love". Select the soonest upcoming event. Choose 2 tickets and continue. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas
# Available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (255, 255, 255)            # page background (dominant white)
status_color = (240, 240, 240)        # top status bar
search_shadow = (238, 238, 238)       # subtle shadow under search
search_fill = (250, 250, 250)         # search field background
search_border = (230, 230, 230)       # search field border
divider_color = (235, 235, 235)       # light separators
nav_border = (225, 225, 225)          # top border of nav bar
nav_fill = (255, 255, 255)            # nav bar fill (white)

# Fill overall background (canvas starts white, but redraw to ensure consistent tone)
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_color)

# Search bar (rounded rectangle) positioned under the status bar.
# Keep margins similar to screenshot: left/right ~40px, height ~148px
search_left = 40
search_top = status_h + 8
search_right = w - 40
search_bottom = search_top + 148
search_radius = 34

# subtle shadow under search (light rounded rectangle offset by a few px)
shadow_offset = 6
draw.rounded_rectangle(
    [(search_left, search_top + shadow_offset),
     (search_right, search_bottom + shadow_offset)],
    radius=search_radius,
    fill=search_shadow
)

# search field fill and border
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=search_fill,
    outline=search_border,
    width=1
)

# thin divider line under the search bar (separates header from content)
line_y = search_bottom + 28
draw.line([(40, line_y), (w - 40, line_y)], fill=divider_color, width=2)

# Large subtle divider between "recent searches" block and suggestions block.
# Place it approximate to screenshot spacing (below recent items).
div_y = 1140  # approximate horizontal separator location
draw.line([(24, div_y), (w - 24, div_y)], fill=divider_color, width=2)

# Additional faint separator a bit lower for another section group
div_y2 = 1508
draw.line([(24, div_y2), (w - 24, div_y2)], fill=divider_color, width=1)

# Bottom navigation bar background (top border + fill area)
nav_top = 2792
draw.rectangle([(0, nav_top), (w, h)], fill=nav_fill)
draw.line([(0, nav_top), (w, nav_top)], fill=nav_border, width=2)

# Subtle top gradient-like band behind the nav (a faint band for depth)
band_h = 8
draw.rectangle([(0, nav_top - band_h), (w, nav_top)], fill=(248, 248, 248))

# Light content area shading blocks to indicate section card backgrounds.
# These are very subtle and wide, not overlapping specific detected icon/text crops.
# Card behind Recent Searches header area
card1_top = search_bottom + 80
card1_bottom = search_bottom + 300
draw.rounded_rectangle(
    [(32, card1_top), (w - 32, card1_bottom)],
    radius=16,
    fill=(255, 255, 255),
    outline=None
)

# Card behind Suggestions section (a larger subtle band)
card2_top = 1400
card2_bottom = 2080
draw.rounded_rectangle(
    [(32, card2_top), (w - 32, card2_bottom)],
    radius=18,
    fill=(255, 255, 255),
    outline=None
)

# Very subtle vertical guides (almost invisible) to suggest content margins
guide_color = (250, 250, 250)
draw.line([(32, status_h), (32, h - 300)], fill=guide_color, width=1)
draw.line([(w - 32, status_h), (w - 32, h - 300)], fill=guide_color, width=1)

# End of background and structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/00_icon_Recent_searches.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/01_icon_The_Phantom_of_the_Opera.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 471), _c1)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/02_icon_Wicked.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 639), _c2)
except Exception:
    pass
layout["Wicked"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/03_icon_Tracking.png
try:
    _c3 = get_crop(3, 288, 168)
    canvas.paste(_c3, (864, 2792), _c3)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 45, 70)
    canvas.paste(_c4, (1154, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1154, 0, 1199, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/05_icon_The_Phantom_of_the_Opera.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 639), _c5)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/06_icon_Suggestions.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 1143), _c6)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 65, 62)
    canvas.paste(_c8, (243, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [243, 3, 308, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/09_icon_5.06_my.png
try:
    _c9 = get_crop(9, 168, 144)
    canvas.paste(_c9, (48, 120), _c9)
except Exception:
    pass
layout["5.06_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/10_icon_Just_Announced_by_My_Performers.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1688), _c10)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (576, 2792), _c11)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/12_icon_Boston_Celtics.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 807), _c12)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 98, 68)
    canvas.paste(_c13, (1216, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1216, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/15_icon_Events_by_My_Performers.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 1520), _c15)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/16_icon_Miami_Dolphins.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 975), _c16)
except Exception:
    pass
layout["Miami_Dolphins"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/17_icon_Account.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (1152, 2792), _c17)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 45, 66)
    canvas.paste(_c18, (1327, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [1327, 2, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 54, 59)
    canvas.paste(_c19, (315, 5), _c19)
except Exception:
    pass
layout["icon_19"] = [315, 5, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/20_icon_Search.png
try:
    _c20 = get_crop(20, 288, 162)
    canvas.paste(_c20, (288, 2792), _c20)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/21_icon_5.06_my.png
try:
    _c21 = get_crop(21, 45, 62)
    canvas.paste(_c21, (187, 2), _c21)
except Exception:
    pass
layout["5.06_my"] = [187, 2, 232, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/22_icon_Performer_event_or_venue.png
try:
    _c22 = get_crop(22, 1032, 144)
    canvas.paste(_c22, (216, 120), _c22)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/25_icon_Wicked.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 807), _c25)
except Exception:
    pass
layout["Wicked"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/26_icon_Miami_Dolphins.png
try:
    _c26 = get_crop(26, 1440, 168)
    canvas.paste(_c26, (0, 1143), _c26)
except Exception:
    pass
layout["Miami_Dolphins"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/27_text_5.06_my.png
try:
    _c27 = get_crop(27, 151, 52)
    canvas.paste(_c27, (21, 9), _c27)
except Exception:
    pass
layout["5.06_my"] = [21, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/28_text_Recent_searches.png
try:
    _c28 = get_crop(28, 168, 144)
    canvas.paste(_c28, (48, 120), _c28)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_03_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-6/29_text_Suggestions.png
try:
    _c29 = get_crop(29, 331, 74)
    canvas.paste(_c29, (40, 1423), _c29)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
