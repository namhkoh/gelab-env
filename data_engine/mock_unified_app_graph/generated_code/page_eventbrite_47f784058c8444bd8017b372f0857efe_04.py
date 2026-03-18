# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_04
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6.png
# step_index: 4/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for Eventbrite-like mobile page
# Canvas (1440x2960) and draw are provided by the environment.

# Colors
bg_color = (247,247,249)          # overall page background (very light gray)
status_bar_color = (190,190,190)  # top status bar (light gray)
divider_color = (226,225,229)     # subtle divider / rule color (lavender gray)
card_shadow_color = (218,218,220) # soft shadow for cards
card_bg = (255,255,255)           # card background (white)
bottom_nav_bg = (250,250,251)     # bottom navigation background

w, h = canvas.size

# Fill page background
draw.rectangle([(0,0),(w,h)], fill=bg_color)

# Status bar area (~top 80px)
status_h = 80
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_color)

# Slight darker top edge for status bar to mimic screenshot subtle border
draw.line([(0,status_h-1),(w,status_h-1)], fill=(176,176,176), width=1)

# Header / search area (below status bar)
# Keep it visually distinct with the same bg but add subtle divider lines.
search_top = status_h
search_bottom = 280
# Draw a very faint inner horizontal rule to separate status and header area
draw.line([(32, search_bottom),(w-32, search_bottom)], fill=divider_color, width=2)

# Thin divider under location / chips area (around where filters are)
chips_div_y = 520
draw.line([(24, chips_div_y),(w-24, chips_div_y)], fill=divider_color, width=1)

# Section title divider above the list (to separate filters/header from content)
list_div_y = chips_div_y + 16
draw.line([(24, list_div_y),(w-24, list_div_y)], fill=(245,244,247), width=6)

# Card 1 background with soft shadow
card_margin_x = 36
card1_top = 560
card1_bottom = 1560
card_radius = 28

# Shadow (drawn as a slightly bigger rounded rect behind the card)
shadow_offset = 10
draw.rounded_rectangle(
    [(card_margin_x+shadow_offset, card1_top+shadow_offset),
     (w-card_margin_x+shadow_offset-0, card1_bottom+shadow_offset)],
    radius=card_radius+2,
    fill=card_shadow_color
)

# Card background
draw.rounded_rectangle(
    [(card_margin_x, card1_top), (w-card_margin_x, card1_bottom)],
    radius=card_radius,
    fill=card_bg
)

# Subtle separator between image area and card text area (approx location)
# The event image will be pasted on top; this rule sits under the image area
img_bottom_hint = 920
draw.line([(card_margin_x+12, img_bottom_hint), (w-card_margin_x-12, img_bottom_hint)],
          fill=(244,243,246), width=2)

# Card 2 background with soft shadow
card2_top = 1760
card2_bottom = 2560

draw.rounded_rectangle(
    [(card_margin_x+shadow_offset, card2_top+shadow_offset),
     (w-card_margin_x+shadow_offset, card2_bottom+shadow_offset)],
    radius=card_radius+2,
    fill=card_shadow_color
)

draw.rounded_rectangle(
    [(card_margin_x, card2_top), (w-card_margin_x, card2_bottom)],
    radius=card_radius,
    fill=card_bg
)

# Divider line between cards list area (a faint rule)
mid_div_y = card2_top - 40
draw.line([(24, mid_div_y), (w-24, mid_div_y)], fill=divider_color, width=1)

# Small decorative pill-shaped placeholder backgrounds for status badges (behind badges)
# These are purely background shapes and will be under any badge text/icons pasted later.
badge_w = 240
badge_h = 48
badge_radius = 24
# First card badge background (light lavender)
badge1_x = card_margin_x + 28
badge1_y = img_bottom_hint - 60
draw.rounded_rectangle(
    [(badge1_x, badge1_y), (badge1_x + badge_w, badge1_y + badge_h)],
    radius=badge_radius, fill=(243,236,245)
)

# Second card badge background (soft pink)
badge2_x = card_margin_x + 28
badge2_y = card2_top + (card_radius // 2)
draw.rounded_rectangle(
    [(badge2_x, badge2_y), (badge2_x + badge_w, badge2_y + badge_h)],
    radius=badge_radius, fill=(252,240,242)
)

# Bottom navigation bar background and top divider
bottom_nav_h = 200
bottom_nav_top = h - bottom_nav_h
draw.rectangle([(0, bottom_nav_top), (w, h)], fill=bottom_nav_bg)
draw.line([(24, bottom_nav_top), (w-24, bottom_nav_top)], fill=divider_color, width=1)

# Subtle shadow above bottom nav to separate from content
draw.line([(24, bottom_nav_top+2), (w-24, bottom_nav_top+2)], fill=(245,244,247), width=2)

# Final subtle vertical guides (left gutter and right gutter) to mimic padding
gutter_x = 48
draw.line([(gutter_x, status_h), (gutter_x, h - bottom_nav_h)], fill=(250,250,251), width=2)
draw.line([(w - gutter_x, status_h), (w - gutter_x, h - bottom_nav_h)], fill=(250,250,251), width=2)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (954, 410), _c0)
except Exception:
    pass
layout["Music"] = [954, 410, 1141, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/01_icon_This_Weekend.png
try:
    _c1 = get_crop(1, 504, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["This_Weekend"] = [438, 410, 942, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/02_icon_Business.png
try:
    _c2 = get_crop(2, 239, 103)
    canvas.paste(_c2, (1153, 410), _c2)
except Exception:
    pass
layout["Business"] = [1153, 410, 1392, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2434), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/05_icon_Afrobeats_Dance_Party_Wizboyy_Performing.png
try:
    _c5 = get_crop(5, 1344, 1194)
    canvas.paste(_c5, (48, 676), _c5)
except Exception:
    pass
layout["Afrobeats_Dance_Party:_Wi"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2434), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/09_icon_7.57.png
try:
    _c9 = get_crop(9, 118, 112)
    canvas.paste(_c9, (58, 115), _c9)
except Exception:
    pass
layout["7.57"] = [58, 115, 176, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 65, 62)
    canvas.paste(_c10, (308, 1), _c10)
except Exception:
    pass
layout["Search_forae"] = [308, 1, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/11_icon_7.57.png
try:
    _c11 = get_crop(11, 58, 63)
    canvas.paste(_c11, (181, 1), _c11)
except Exception:
    pass
layout["7.57"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/12_icon_7.57.png
try:
    _c12 = get_crop(12, 58, 65)
    canvas.paste(_c12, (114, 0), _c12)
except Exception:
    pass
layout["7.57"] = [114, 0, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 51, 63)
    canvas.paste(_c13, (247, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [247, 1, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 83, 61)
    canvas.paste(_c14, (1210, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1210, 0, 1293, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/15_icon_Anthony_Falco_-.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["Anthony_Falco_-"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 58, 61)
    canvas.paste(_c16, (1317, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1317, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/17_icon_Wood_Fired_Master_Class.png
try:
    _c17 = get_crop(17, 1344, 898)
    canvas.paste(_c17, (48, 1918), _c17)
except Exception:
    pass
layout["Wood_Fired_Master_Class"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/18_icon_San_Francisco.png
try:
    _c18 = get_crop(18, 536, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/20_icon_Few_tickets_left.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Few_tickets_left"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/21_icon_Afrobeats_Dance_Party_Wizboyy_Performing.png
try:
    _c21 = get_crop(21, 1344, 1194)
    canvas.paste(_c21, (48, 676), _c21)
except Exception:
    pass
layout["Afrobeats_Dance_Party:_Wi"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/22_icon_Afrobeats_Dance_Party_Wizboyy_Performing.png
try:
    _c22 = get_crop(22, 1344, 1194)
    canvas.paste(_c22, (48, 676), _c22)
except Exception:
    pass
layout["Afrobeats_Dance_Party:_Wi"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/23_icon_Search_forae.png
try:
    _c23 = get_crop(23, 48, 62)
    canvas.paste(_c23, (384, 2), _c23)
except Exception:
    pass
layout["Search_forae"] = [384, 2, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/24_icon_7.57.png
try:
    _c24 = get_crop(24, 96, 64)
    canvas.paste(_c24, (12, 0), _c24)
except Exception:
    pass
layout["7.57"] = [12, 0, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/25_icon_Wood_Fired_Master_Class.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Wood_Fired_Master_Class"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/26_icon_Anthony_Falco_-.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Anthony_Falco_-"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 41, 62)
    canvas.paste(_c27, (1273, 0), _c27)
except Exception:
    pass
layout["icon_27"] = [1273, 0, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/29_text_2_921_events.png
try:
    _c29 = get_crop(29, 372, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["2,921_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_04_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-6/30_clickable_Tickets.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (864, 2804), _c30)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]
