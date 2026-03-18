# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_11
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13.png
# step_index: 11/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI mockup.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

w, h = canvas.size

# Colors
bg_offwhite = (250, 248, 253)      # subtle off-white page background
status_bar_gray = (220, 220, 220)  # top status bar
hero_start = (255, 77, 170)        # pink
hero_end = (143, 58, 255)          # purple
hero_fade_top = (255, 255, 255)    # fade into white
divider = (236, 237, 242)          # light divider lines
card_bg = (247, 248, 251)          # organizer card background
card_border = (226, 227, 232)      # organizer card border
shadow = (230, 228, 234)           # subtle shadow
pill_gray = (230, 232, 237)
pill_active = (82, 26, 116)        # dark purple active pill

# Fill overall background
draw.rectangle([0, 0, w, h], fill=bg_offwhite)

# Status bar (top)
status_h = 88
draw.rectangle([0, 0, w, status_h], fill=status_bar_gray)

# Hero image area (gradient)
hero_top = status_h
hero_h = 420
hero_bottom = hero_top + hero_h
for i in range(hero_h):
    t = i / max(1, hero_h - 1)
    r = int(hero_start[0] + (hero_end[0] - hero_start[0]) * t)
    g = int(hero_start[1] + (hero_end[1] - hero_start[1]) * t)
    b = int(hero_start[2] + (hero_end[2] - hero_start[2]) * t)
    draw.rectangle([0, hero_top + i, w, hero_top + i + 1], fill=(r, g, b))

# Soft bottom fade of hero to link into white content
fade_height = 48
for i in range(fade_height):
    t = i / max(1, fade_height - 1)
    r = int((1 - t) * hero_end[0] + t * hero_fade_top[0])
    g = int((1 - t) * hero_end[1] + t * hero_fade_top[1])
    b = int((1 - t) * hero_end[2] + t * hero_fade_top[2])
    draw.rectangle([0, hero_bottom - fade_height + i, w, hero_bottom - fade_height + i + 1], fill=(r, g, b))

# Divider line below hero
draw.rectangle([48, hero_bottom + 6, w - 48, hero_bottom + 8], fill=divider)

# Major content area is left as the off-white background (text and icons pasted on top)

# Organizer/host card (rounded rectangle)
card_x1 = 48
card_x2 = w - 48
card_y1 = 1180
card_y2 = 1344
card_radius = 28
# subtle shadow behind card
draw.rounded_rectangle([card_x1 - 6, card_y1 + 6, card_x2 + 6, card_y2 + 8], radius=card_radius + 2, fill=shadow)
# card fill and border
draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=card_radius, fill=card_bg, outline=card_border, width=2)

# Thin separator under organizer card
sep_y = card_y2 + 60
draw.rectangle([48, sep_y, w - 48, sep_y + 1], fill=divider)

# Small informational icon row area (left empty for pasted icons/text) -- draw subtle guiding background strip
info_strip_y1 = card_y2 + 28
info_strip_y2 = info_strip_y1 + 52
draw.rectangle([48, info_strip_y1, w - 48, info_strip_y2], fill=bg_offwhite)

# Light horizontal rule above "Select date and time"
select_line_y = 2026
draw.rectangle([48, select_line_y, w - 48, select_line_y + 2], fill=divider)

# "Select date and time" pill indicators row (just background pills)
pills_top = 1760
pills_left = 56
pill_w = 140
pill_h = 18
pill_gap = 22
# draw a line of light pills
for i in range(7):
    x1 = pills_left + i * (pill_w + pill_gap)
    x2 = x1 + pill_w
    y1 = pills_top
    y2 = y1 + pill_h
    # keep pills within margins
    if x1 < w - 56:
        draw.rounded_rectangle([x1, y1, min(x2, w - 56), y2], radius=9, fill=pill_gray)
# active pill (shorter, dark)
active_x = pills_left
draw.rounded_rectangle([active_x, pills_top - 8, active_x + 96, pills_top + pill_h - 8], radius=9, fill=pill_active)

# Subtle section divider above reservation area (do not draw reservation content)
bottom_section_div_y = 2160
draw.rectangle([0, bottom_section_div_y, w, bottom_section_div_y + 2], fill=divider)

# Leave lower region blank (elements such as reservation card/button will be pasted on top)

# Final subtle top and bottom edge lines to match screenshot subtleness
draw.rectangle([0, 0, w, 1], fill=divider)
draw.rectangle([0, h - 1, w, h], fill=divider)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1258), _c0)
except Exception:
    pass
layout["Following"] = [946, 1258, 1344, 1402]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/01_icon_9.20.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (36, 108), _c1)
except Exception:
    pass
layout["9.20"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/04_icon_Reserve_a.png
try:
    _c4 = get_crop(4, 1440, 753)
    canvas.paste(_c4, (0, 2207), _c4)
except Exception:
    pass
layout["Reserve_a"] = [0, 2207, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/05_icon_Read_more.png
try:
    _c5 = get_crop(5, 239, 62)
    canvas.paste(_c5, (1089, 2492), _c5)
except Exception:
    pass
layout["Read_more"] = [1089, 2492, 1328, 2554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 48, 68)
    canvas.paste(_c6, (1154, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1154, 2, 1202, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/07_icon_SABOR.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1116, 108), _c7)
except Exception:
    pass
layout["SABOR"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/08_icon_03.10.png
try:
    _c8 = get_crop(8, 55, 59)
    canvas.paste(_c8, (314, 3), _c8)
except Exception:
    pass
layout["03.10"] = [314, 3, 369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 44, 64)
    canvas.paste(_c9, (1328, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1328, 3, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 49, 58)
    canvas.paste(_c10, (383, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [383, 4, 432, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/11_icon_12_00_PM_Complimentary_Reservation.png
try:
    _c11 = get_crop(11, 1440, 753)
    canvas.paste(_c11, (0, 2207), _c11)
except Exception:
    pass
layout["12:00_PM_Complimentary_Re"] = [0, 2207, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/12_icon_03.10.png
try:
    _c12 = get_crop(12, 56, 60)
    canvas.paste(_c12, (247, 2), _c12)
except Exception:
    pass
layout["03.10"] = [247, 2, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/13_icon_03.10.png
try:
    _c13 = get_crop(13, 51, 62)
    canvas.paste(_c13, (184, 1), _c13)
except Exception:
    pass
layout["03.10"] = [184, 1, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 95, 67)
    canvas.paste(_c14, (1213, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1213, 1, 1308, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/15_icon_9.20.png
try:
    _c15 = get_crop(15, 55, 62)
    canvas.paste(_c15, (115, 2), _c15)
except Exception:
    pass
layout["9.20"] = [115, 2, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/16_text_9.20.png
try:
    _c16 = get_crop(16, 91, 41)
    canvas.paste(_c16, (20, 17), _c16)
except Exception:
    pass
layout["9.20"] = [20, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/17_text_Sunday_March_24.png
try:
    _c17 = get_crop(17, 460, 73)
    canvas.paste(_c17, (38, 760), _c17)
except Exception:
    pass
layout["Sunday;_March_24"] = [38, 760, 498, 833]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/18_text_12_00_PM.png
try:
    _c18 = get_crop(18, 240, 56)
    canvas.paste(_c18, (527, 766), _c18)
except Exception:
    pass
layout["12:00_PM"] = [527, 766, 767, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/19_text_MAMAZUL_-_Sabor.png
try:
    _c19 = get_crop(19, 207, 144)
    canvas.paste(_c19, (288, 1218), _c19)
except Exception:
    pass
layout["MAMAZUL_-_Sabor"] = [288, 1218, 495, 1362]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/20_text_LIVEE_Show_Latin.png
try:
    _c20 = get_crop(20, 207, 144)
    canvas.paste(_c20, (288, 1218), _c20)
except Exception:
    pass
layout["LIVEE_Show_+_Latin_&"] = [288, 1218, 495, 1362]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/21_text_Party.png
try:
    _c21 = get_crop(21, 203, 97)
    canvas.paste(_c21, (37, 1074), _c21)
except Exception:
    pass
layout["Party"] = [37, 1074, 240, 1171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/22_text_iBoatNYC.png
try:
    _c22 = get_crop(22, 207, 144)
    canvas.paste(_c22, (288, 1218), _c22)
except Exception:
    pass
layout["iBoatNYC"] = [288, 1218, 495, 1362]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/23_text_47.6k_Followers.png
try:
    _c23 = get_crop(23, 207, 144)
    canvas.paste(_c23, (288, 1218), _c23)
except Exception:
    pass
layout["47.6k_Followers"] = [288, 1218, 495, 1362]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/24_text_Mamazul.png
try:
    _c24 = get_crop(24, 203, 54)
    canvas.paste(_c24, (141, 1533), _c24)
except Exception:
    pass
layout["Mamazul"] = [141, 1533, 344, 1587]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/25_text_5_hrs.png
try:
    _c25 = get_crop(25, 112, 50)
    canvas.paste(_c25, (141, 1642), _c25)
except Exception:
    pass
layout["5_hrs"] = [141, 1642, 253, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/26_text_Refund_policy.png
try:
    _c26 = get_crop(26, 299, 64)
    canvas.paste(_c26, (138, 1747), _c26)
except Exception:
    pass
layout["Refund_policy"] = [138, 1747, 437, 1811]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/27_text_The_organizer_will_review_refund_request.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1485), _c27)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1485, 1392, 1629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/28_text_Select_date_and_time.png
try:
    _c28 = get_crop(28, 569, 63)
    canvas.paste(_c28, (41, 2046), _c28)
except Exception:
    pass
layout["Select_date_and_time"] = [41, 2046, 610, 2109]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_11_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-13/29_clickable_Organizer_profile_picture.png
try:
    _c29 = get_crop(29, 144, 144)
    canvas.paste(_c29, (96, 1257), _c29)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1257, 240, 1401]
