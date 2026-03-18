# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_09
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11.png
# step_index: 9/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL Image 1440x2960, draw: ImageDraw)
w, h = canvas.size

# Colors
bg = "#f7f8fb"           # page background (very light)
status_bg = "#d1d1d1"    # status bar background (light gray)
divider = "#e3e3e6"      # thin dividers
card_shadow = "#e9ebef"  # subtle card shadow
card_bg = "#ffffff"      # card background
search_bg = "#ffffff"    # search field background
bottom_nav_bg = "#ffffff"  # bottom navigation background
muted = "#f2f5f7"

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg)

# Status bar area at top (~80px)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=status_bg)
# subtle bottom border under status bar
draw.line([(0, status_h), (w, status_h)], fill=divider, width=1)

# Search/header area background (rounded search field backdrop)
search_x1, search_y1 = 48, 72
search_x2, search_y2 = 48 + 1344, 72 + 190  # matches detected search field bbox
search_radius = 20
# Slight shadow for search area
shadow_offset = 4
draw.rounded_rectangle(
    [(search_x1, search_y1 + shadow_offset), (search_x2, search_y2 + shadow_offset)],
    radius=search_radius, fill=card_shadow
)
draw.rounded_rectangle(
    [(search_x1, search_y1), (search_x2, search_y2)],
    radius=search_radius, fill=search_bg, outline=divider, width=1
)

# Thin separator below header/search area
sep_y = search_y2 + 18
draw.line([(48, sep_y), (w - 48, sep_y)], fill=divider, width=1)

# Filter chips row area (subtle background band behind chips)
chips_top = 360
chips_bottom = 480
draw.rectangle([(0, chips_top), (w, chips_bottom)], fill=bg)
# subtle line under chips
draw.line([(48, chips_bottom), (w - 48, chips_bottom)], fill=divider, width=1)

# First event card background (rounded rectangle with subtle shadow)
card1_x1, card1_y1 = 48, 676
card1_w, card1_h = 1344, 1175
card1_x2, card1_y2 = card1_x1 + card1_w, card1_y1 + card1_h
card_radius = 24
# shadow
draw.rounded_rectangle(
    [(card1_x1 + 6, card1_y1 + 8), (card1_x2 + 6, card1_y2 + 8)],
    radius=card_radius + 2, fill=card_shadow
)
# card background
draw.rounded_rectangle(
    [(card1_x1, card1_y1), (card1_x2, card1_y2)],
    radius=card_radius, fill=card_bg, outline=divider, width=1
)

# Separator line between image area and card content (approximate)
sep1 = card1_y1 + int(card1_h * 0.45)
draw.line([(card1_x1 + 24, sep1), (card1_x2 - 24, sep1)], fill=muted, width=1)

# Second event card background (rounded rectangle with subtle shadow)
card2_x1, card2_y1 = 48, 1899
card2_w, card2_h = 1344, 917
card2_x2, card2_y2 = card2_x1 + card2_w, card2_y1 + card2_h
# shadow
draw.rounded_rectangle(
    [(card2_x1 + 6, card2_y1 + 8), (card2_x2 + 6, card2_y2 + 8)],
    radius=card_radius + 2, fill=card_shadow
)
# card background
draw.rounded_rectangle(
    [(card2_x1, card2_y1), (card2_x2, card2_y2)],
    radius=card_radius, fill=card_bg, outline=divider, width=1
)

# Small separator between stacked sections
between_y = card1_y2 + 24
draw.line([(48, between_y), (w - 48, between_y)], fill=divider, width=1)

# Bottom navigation bar background and top border
nav_h = 120
nav_y1 = h - nav_h
draw.rectangle([(0, nav_y1), (w, h)], fill=bottom_nav_bg)
draw.line([(0, nav_y1), (w, nav_y1)], fill=divider, width=1)

# Add faint left/right margins lines to mirror content padding
padding_x = 48
draw.line([(padding_x, nav_y1), (padding_x, h)], fill=muted, width=1)
draw.line([(w - padding_x, nav_y1), (w - padding_x, h)], fill=muted, width=1)

# Final subtle horizontal separators across page at logical breaks
for y in (search_y2 + 60, card1_y2 + 12, card2_y2 + 12):
    draw.line([(48, y), (w - 48, y)], fill=divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 432, 103)
    canvas.paste(_c0, (438, 410), _c0)
except Exception:
    pass
layout["Tomorrow"] = [438, 410, 870, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 103)
    canvas.paste(_c1, (882, 410), _c1)
except Exception:
    pass
layout["Music"] = [882, 410, 1069, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1081, 410), _c2)
except Exception:
    pass
layout["Business"] = [1081, 410, 1322, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/04_icon_Business.png
try:
    _c4 = get_crop(4, 93, 108)
    canvas.paste(_c4, (1329, 408), _c4)
except Exception:
    pass
layout["Business"] = [1329, 408, 1422, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/05_icon_Manthly_Contribution_USD4.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Manthly_Contribution;_USD"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/07_icon_8.07.png
try:
    _c7 = get_crop(7, 124, 114)
    canvas.paste(_c7, (55, 114), _c7)
except Exception:
    pass
layout["8.07"] = [55, 114, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 65)
    canvas.paste(_c9, (1152, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1152, 0, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 68, 63)
    canvas.paste(_c10, (307, 0), _c10)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/11_icon_8.07.png
try:
    _c11 = get_crop(11, 59, 64)
    canvas.paste(_c11, (181, 0), _c11)
except Exception:
    pass
layout["8.07"] = [181, 0, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 83, 62)
    canvas.paste(_c12, (1213, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1213, 0, 1296, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/13_icon_8.07.png
try:
    _c13 = get_crop(13, 61, 66)
    canvas.paste(_c13, (113, 0), _c13)
except Exception:
    pass
layout["8.07"] = [113, 0, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 64)
    canvas.paste(_c14, (247, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 0, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1236, 1192), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 58, 60)
    canvas.paste(_c16, (1316, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1316, 0, 1374, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/17_icon_Chicago.png
try:
    _c17 = get_crop(17, 417, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/19_icon_Ticket_sales_end_soon.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/20_icon_Spiritual_Leadership_Develpoment_YOU_ARE.png
try:
    _c20 = get_crop(20, 1344, 1175)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["Spiritual_Leadership_Deve"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/21_icon_Ecstatic_Dance_Full_Moon_Fusion_Cacao.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Ecstatic_Dance_+_Full_Moo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/22_icon_Ecstatic_Dance_Full_Moon_Fusion_Cacao.png
try:
    _c22 = get_crop(22, 1344, 917)
    canvas.paste(_c22, (48, 1899), _c22)
except Exception:
    pass
layout["Ecstatic_Dance_+_Full_Moo"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/23_icon_Search_forae.png
try:
    _c23 = get_crop(23, 51, 62)
    canvas.paste(_c23, (383, 2), _c23)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/24_icon_Ecstatic_Dance_Full_Moon_Fusion_Cacao.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["Ecstatic_Dance_+_Full_Moo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/25_icon_Ecstatic_Dance_Full_Moon_Fusion_Cacao.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Ecstatic_Dance_+_Full_Moo"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 255, 65)
    canvas.paste(_c26, (75, 1743), _c26)
except Exception:
    pass
layout["Promoted"] = [75, 1743, 330, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 41, 61)
    canvas.paste(_c27, (1273, 0), _c27)
except Exception:
    pass
layout["icon_27"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/28_icon_8.07.png
try:
    _c28 = get_crop(28, 151, 63)
    canvas.paste(_c28, (7, 1), _c28)
except Exception:
    pass
layout["8.07"] = [7, 1, 158, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/29_icon_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/30_text_843_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["843_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/31_text_LLADERSHIP_DEVELOPMENT_SESSIONS.png
try:
    _c31 = get_crop(31, 1344, 1175)
    canvas.paste(_c31, (48, 676), _c31)
except Exception:
    pass
layout["LLADERSHIP_DEVELOPMENT_SE"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/32_text_YOUARi_MIIE_SOURcE_0i_YOUR_DLSTINY.png
try:
    _c32 = get_crop(32, 1344, 1175)
    canvas.paste(_c32, (48, 676), _c32)
except Exception:
    pass
layout["YOUARi_MIIE_SOURcE_0i_YOU"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/33_text_Develop_your_leadership_skills_asyou_cmb.png
try:
    _c33 = get_crop(33, 1344, 1175)
    canvas.paste(_c33, (48, 676), _c33)
except Exception:
    pass
layout["'Develop_your_leadership_"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/34_text_Online.png
try:
    _c34 = get_crop(34, 129, 45)
    canvas.paste(_c34, (91, 1687), _c34)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_09_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-11/35_text_elem_35.png
try:
    _c35 = get_crop(35, 89, 30)
    canvas.paste(_c35, (104, 2779), _c35)
except Exception:
    pass
layout["_+"] = [104, 2779, 193, 2809]
