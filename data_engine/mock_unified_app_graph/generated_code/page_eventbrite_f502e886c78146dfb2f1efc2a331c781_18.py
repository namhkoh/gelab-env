# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_18
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20.png
# step_index: 18/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile UI page.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (250, 251, 252)        # very light off-white background
status_bar_color = (200, 200, 200)  # light gray status bar
toolbar_color = (255, 255, 255)     # white toolbar/background for header
divider_color = (230, 231, 233)     # subtle divider lines
card_shadow = (224, 227, 230)       # shadow for cards
card_bg = (255, 255, 255)           # card background (white)
bottom_nav_bg = (255, 255, 255)     # bottom nav background

# Utility for rounded rect (delegates to ImageDraw.rounded_rectangle)
def rr(rect, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (top ~96px)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Thin subtle bottom edge for status bar (to separate from header)
draw.line([(0, status_h), (w, status_h)], fill=divider_color, width=1)

# Main toolbar/header area (search bar area) below status bar
toolbar_top = status_h
toolbar_h = 160
toolbar_bottom = toolbar_top + toolbar_h
draw.rectangle([(0, toolbar_top), (w, toolbar_bottom)], fill=toolbar_color)

# Divider under toolbar
draw.line([(24, toolbar_bottom), (w-24, toolbar_bottom)], fill=divider_color, width=1)

# Horizontal subtle separator line between header and filter area (lower)
filter_separator_y = 420
draw.line([(24, filter_separator_y), (w-24, filter_separator_y)], fill=divider_color, width=1)

# Card container backgrounds (rounded rectangles) for event items.
# First card (matches image card area but will be overlaid by pasted content)
card1_x = 48
card1_y = 676
card1_w = 1344
card1_h = 1012
card1_rect = (card1_x-6, card1_y-6, card1_x + card1_w + 6, card1_y + card1_h + 6)
rr(card1_rect, radius=28, fill=card_shadow, outline=None)

card1_inner = (card1_x, card1_y, card1_x + card1_w, card1_y + card1_h)
rr(card1_inner, radius=20, fill=card_bg, outline=divider_color, width=1)

# Space below image area inside the card (content area) - keep background only (no text)
# We'll leave it white (card_bg) so that pasted text/icons will appear correctly.

# Second card (another event card lower on the page)
card2_x = 48
card2_y = 1736
card2_w = 1344
card2_h = 1029
card2_rect = (card2_x-6, card2_y-6, card2_x + card2_w + 6, card2_y + card2_h + 6)
rr(card2_rect, radius=28, fill=card_shadow, outline=None)

card2_inner = (card2_x, card2_y, card2_x + card2_w, card2_y + card2_h)
rr(card2_inner, radius=20, fill=card_bg, outline=divider_color, width=1)

# Thin separators between list items (subtle)
sep_x0 = 24
sep_x1 = w - 24
# Separator just above first card (under the "772 events" area)
sep_y_first = card1_y - 36
draw.line([(sep_x0, sep_y_first), (sep_x1, sep_y_first)], fill=divider_color, width=1)
# Separator between first and second card (slightly below first card)
sep_y_between = card1_y + card1_h + 24
draw.line([(sep_x0, sep_y_between), (sep_x1, sep_y_between)], fill=divider_color, width=1)

# Bottom navigation bar area (reserve space; top divider)
bottom_nav_h = 156
bottom_nav_top = h - bottom_nav_h
draw.rectangle([(0, bottom_nav_top), (w, h)], fill=bottom_nav_bg)
# Divider line above nav
draw.line([(0, bottom_nav_top), (w, bottom_nav_top)], fill=divider_color, width=1)

# A subtle elevated center button background in the nav (placeholder background only)
# (Do not draw icons or labels — just the structural rounded background)
center_btn_w = 84
center_btn_h = 12  # small accent bar to hint center area (structural only)
center_btn_x = w // 2 - center_btn_w // 2
center_btn_y = bottom_nav_top + 18
draw.rounded_rectangle([(center_btn_x, center_btn_y),
                        (center_btn_x + center_btn_w, center_btn_y + center_btn_h)],
                       radius=6, fill=divider_color)

# Optional: subtle left edge vertical guide for filter column (do not draw icons/text)
left_margin_x = 24
draw.line([(left_margin_x, toolbar_bottom + 8), (left_margin_x, bottom_nav_top - 8)], fill=(245,245,246), width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (1111, 410), _c0)
except Exception:
    pass
layout["Music"] = [1111, 410, 1298, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/01_icon_Apr_30_-_May_03_2024.png
try:
    _c1 = get_crop(1, 661, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Apr_30_-_May_03,_2024"] = [438, 410, 1099, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/02_icon_Favorite_button.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1092, 2252), _c2)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 2252), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/07_icon_One_Night_in_Memphis_Tickets.png
try:
    _c7 = get_crop(7, 1344, 1012)
    canvas.paste(_c7, (48, 676), _c7)
except Exception:
    pass
layout["One_Night_in_Memphis_Tick"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/08_icon_Close_current_screen.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 49, 65)
    canvas.paste(_c9, (1153, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1153, 0, 1202, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 64, 63)
    canvas.paste(_c10, (309, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [309, 0, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/11_icon_The_Village_at_San_Antonio_Center_San_An.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (576, 2804), _c11)
except Exception:
    pass
layout["The_Village_at_San_Antoni"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 98, 65)
    canvas.paste(_c12, (1213, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1213, 0, 1311, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/13_icon_7.19.png
try:
    _c13 = get_crop(13, 56, 63)
    canvas.paste(_c13, (182, 0), _c13)
except Exception:
    pass
layout["7.19"] = [182, 0, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 47, 61)
    canvas.paste(_c14, (251, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [251, 1, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/15_icon_7.19.png
try:
    _c15 = get_crop(15, 118, 115)
    canvas.paste(_c15, (58, 113), _c15)
except Exception:
    pass
layout["7.19"] = [58, 113, 176, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/16_icon_7.19.png
try:
    _c16 = get_crop(16, 58, 64)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["7.19"] = [115, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/17_icon_Bu.png
try:
    _c17 = get_crop(17, 96, 111)
    canvas.paste(_c17, (1306, 406), _c17)
except Exception:
    pass
layout["Bu"] = [1306, 406, 1402, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 53, 64)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/19_icon_Live_Music_First_Thursdays.png
try:
    _c19 = get_crop(19, 1344, 1029)
    canvas.paste(_c19, (48, 1736), _c19)
except Exception:
    pass
layout["Live_Music,_First_Thursda"] = [48, 1736, 1392, 2765]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/20_icon_Music_Festival.png
try:
    _c20 = get_crop(20, 47, 62)
    canvas.paste(_c20, (384, 2), _c20)
except Exception:
    pass
layout["Music_Festival"] = [384, 2, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/21_icon_Music_Festival.png
try:
    _c21 = get_crop(21, 1344, 191)
    canvas.paste(_c21, (48, 72), _c21)
except Exception:
    pass
layout["Music_Festival"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/22_icon_The_Village_at_San_Antonio_Center_San_An.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["The_Village_at_San_Antoni"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/23_icon_The_Village_at_San_Antonio_Center_San_An.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["The_Village_at_San_Antoni"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/24_icon_San_Francisco.png
try:
    _c24 = get_crop(24, 536, 144)
    canvas.paste(_c24, (0, 259), _c24)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/25_icon_2_._6_00_PM_PDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (288, 2804), _c25)
except Exception:
    pass
layout["2_._6:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/26_icon_7.19.png
try:
    _c26 = get_crop(26, 137, 63)
    canvas.paste(_c26, (8, 0), _c26)
except Exception:
    pass
layout["7.19"] = [8, 0, 145, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/27_icon_Promoted.png
try:
    _c27 = get_crop(27, 38, 55)
    canvas.paste(_c27, (287, 1586), _c27)
except Exception:
    pass
layout["Promoted"] = [287, 1586, 325, 1641]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/28_text_772_events.png
try:
    _c28 = get_crop(28, 372, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["772_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_18_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-20/29_clickable_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
