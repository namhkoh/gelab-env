# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_04
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6.png
# step_index: 4/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI layout for Eventbrite-like page
w, h = canvas.size

# Background fill (dominant page color - soft white)
draw.rectangle([0, 0, w, h], fill="#FFFFFF")

# Status bar (top ~80px) - light gray
status_h = 80
draw.rectangle([0, 0, w, status_h], fill="#CFCFCF")

# Subtle separator under status bar (thin)
draw.line([(0, status_h), (w, status_h)], fill="#BFBFBF", width=1)

# Header / toolbar area (below status bar)
header_y0 = status_h
header_y1 = 160
draw.rectangle([0, header_y0, w, header_y1], fill="#FFFFFF")
# Bottom divider of header
draw.line([(24, header_y1), (w-24, header_y1)], fill="#E6E6E6", width=2)

# Light shadow under header for depth
draw.line([(0, header_y1+2), (w, header_y1+2)], fill="#F2F2F2", width=1)

# Thin global horizontal separator (between toolbar and filters/content)
sep_y = 320
draw.line([(24, sep_y), (w-24, sep_y)], fill="#F0F0F0", width=1)

# Card 1 (first event) background + subtle shadow
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1194
# shadow (offset)
shadow_offset = 8
draw.rectangle(
    [card1_x + shadow_offset, card1_y + shadow_offset, card1_x + card1_w + shadow_offset, card1_y + card1_h + shadow_offset],
    fill="#EFEFEF"
)
# card background with rounded corners and light border
r = 28
draw.rounded_rectangle(
    [card1_x, card1_y, card1_x + card1_w, card1_y + card1_h],
    radius=r, fill="#FFFFFF", outline="#E9E9E9", width=2
)

# Separator between card image area and the card body (subtle)
# approximate position a bit below the top third of the card to suggest image area
img_split_y = card1_y + int(card1_h * 0.38)
draw.line([(card1_x + 24, img_split_y), (card1_x + card1_w - 24, img_split_y)], fill="#F3F3F3", width=1)

# Card 2 (second event) background + subtle shadow
card2_x, card2_y = 48, 1918
card2_w, card2_h = 1344, 898
draw.rectangle(
    [card2_x + shadow_offset, card2_y + shadow_offset, card2_x + card2_w + shadow_offset, card2_y + card2_h + shadow_offset],
    fill="#EFEFEF"
)
draw.rounded_rectangle(
    [card2_x, card2_y, card2_x + card2_w, card2_y + card2_h],
    radius=r, fill="#FFFFFF", outline="#E9E9E9", width=2
)
# image/body split for second card
img2_split_y = card2_y + int(card2_h * 0.47)
draw.line([(card2_x + 24, img2_split_y), (card2_x + card2_w - 24, img2_split_y)], fill="#F3F3F3", width=1)

# Light section separators for list flow
# Separator below first card
sep1_y = card1_y + card1_h + 28
draw.line([(24, sep1_y), (w - 24, sep1_y)], fill="#F2F2F2", width=1)
# Separator below second card
sep2_y = card2_y + card2_h + 28
draw.line([(24, sep2_y), (w - 24, sep2_y)], fill="#F2F2F2", width=1)

# Bottom navigation bar area (approx 120px high)
nav_h = 120
nav_y0 = h - nav_h
draw.rectangle([0, nav_y0, w, h], fill="#FFFFFF")
# Top border of nav
draw.line([(0, nav_y0), (w, nav_y0)], fill="#E2E2E2", width=2)

# Subtle center indicator line above nav (visual groove)
draw.line([(24, nav_y0 + 1), (w - 24, nav_y0 + 1)], fill="#F6F6F6", width=1)

# Decorative left/right page margins (very subtle vertical guides)
# these are purely structural and very faint
draw.line([(24, header_y1 + 8), (24, h - nav_h - 8)], fill="#FBFBFB", width=1)
draw.line([(w - 24, header_y1 + 8), (w - 24, h - nav_h - 8)], fill="#FBFBFB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/05_icon_RrD_CARFet_I_WE_RFTET.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["RrD_CARFet_I_WE_*_RFTET"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2434), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2434), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/09_icon_Sports.png
try:
    _c9 = get_crop(9, 1344, 191)
    canvas.paste(_c9, (48, 72), _c9)
except Exception:
    pass
layout["Sports"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/10_icon_4.35.png
try:
    _c10 = get_crop(10, 123, 112)
    canvas.paste(_c10, (56, 114), _c10)
except Exception:
    pass
layout["4.35"] = [56, 114, 179, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/11_icon_Foo.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 96), _c11)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/12_icon_Sports.png
try:
    _c12 = get_crop(12, 70, 64)
    canvas.paste(_c12, (307, 0), _c12)
except Exception:
    pass
layout["Sports"] = [307, 0, 377, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/13_icon_Sports.png
try:
    _c13 = get_crop(13, 54, 64)
    canvas.paste(_c13, (246, 0), _c13)
except Exception:
    pass
layout["Sports"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/14_icon_4.35.png
try:
    _c14 = get_crop(14, 61, 63)
    canvas.paste(_c14, (181, 0), _c14)
except Exception:
    pass
layout["4.35"] = [181, 0, 242, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 105, 61)
    canvas.paste(_c15, (1205, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1205, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/16_icon_4.35.png
try:
    _c16 = get_crop(16, 62, 65)
    canvas.paste(_c16, (114, 0), _c16)
except Exception:
    pass
layout["4.35"] = [114, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 60, 60)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1378, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/18_icon_The_Masquerade_Ball_at_The_Historic_Clif.png
try:
    _c18 = get_crop(18, 1344, 1194)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["The_Masquerade_Ball_at_Th"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/19_icon_San_Francisco.png
try:
    _c19 = get_crop(19, 536, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/20_icon_THE_CLIFT.png
try:
    _c20 = get_crop(20, 1344, 1194)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["THE_CLIFT"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 51, 61)
    canvas.paste(_c21, (384, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [384, 3, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/22_icon_lun_20_11.0_AAA_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["lun_20_,_11.0_AAA_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/23_icon_Online_SEO_Course.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Online_SEO_Course"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/24_icon_Master_Google_s_New_Search_Generative.png
try:
    _c24 = get_crop(24, 1344, 898)
    canvas.paste(_c24, (48, 1918), _c24)
except Exception:
    pass
layout["Master_Google's_New_Searc"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/25_icon_Online_SEO_Course.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Online_SEO_Course"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 246, 61)
    canvas.paste(_c26, (83, 1765), _c26)
except Exception:
    pass
layout["Promoted"] = [83, 1765, 329, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/27_icon_4.35.png
try:
    _c27 = get_crop(27, 125, 63)
    canvas.paste(_c27, (8, 0), _c27)
except Exception:
    pass
layout["4.35"] = [8, 0, 133, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/28_icon_Tu.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Tu"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/29_icon_The_Clift_Royal_Sonesta_Hotel.png
try:
    _c29 = get_crop(29, 46, 58)
    canvas.paste(_c29, (283, 1766), _c29)
except Exception:
    pass
layout["The_Clift_Royal_Sonesta_H"] = [283, 1766, 329, 1824]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/30_text_1_376_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["1,376_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/31_text_STTurdrySmrtm.png
try:
    _c31 = get_crop(31, 268, 27)
    canvas.paste(_c31, (585, 675), _c31)
except Exception:
    pass
layout["STTurdrySmrtm"] = [585, 675, 853, 702]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/32_text_M_FRALCISCO_6THANNUAL.png
try:
    _c32 = get_crop(32, 334, 36)
    canvas.paste(_c32, (552, 708), _c32)
except Exception:
    pass
layout["M'_FRALCISCO_$_6THANNUAL"] = [552, 708, 886, 744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/33_text_MASOBERADE.png
try:
    _c33 = get_crop(33, 1344, 1194)
    canvas.paste(_c33, (48, 676), _c33)
except Exception:
    pass
layout["MASOBERADE"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/34_text_The_Clift_Royal_Sonesta_Hotel.png
try:
    _c34 = get_crop(34, 540, 56)
    canvas.paste(_c34, (93, 1704), _c34)
except Exception:
    pass
layout["The_Clift_Royal_Sonesta_H"] = [93, 1704, 633, 1760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_04_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-6/35_clickable_More.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (1152, 2804), _c35)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
