# page_id: page_eventbrite_f56b9f3bf9ef483cbd9847af47d34434_02
# screenshot: 2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4.png
# step_index: 2/8
# task: Open Eventbrite. Look up "Gardening" events. Filter by events happening this week. Select the first event from the results. Follow the organizer and where is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw UI background and structure for 1440x2960 canvas
# (assumes `canvas` is a PIL Image and `draw` is an ImageDraw object,
#  and font_sm/font_md/font_lg/font_xl are available)

w, h = canvas.size

# Colors
COLOR_BG = (255, 255, 255)           # main background (white)
COLOR_STATUS = (200, 200, 200)       # status bar gray
COLOR_HEADER = (255, 255, 255)       # header background (white)
COLOR_PRIMARY = (27, 71, 255)        # bright accent blue for underline
COLOR_DIV = (230, 230, 235)          # light divider lines
COLOR_CARD = (252, 252, 253)         # very subtle card background
COLOR_NAV_BORDER = (220, 220, 224)   # nav top border

# Clear canvas to background (defensive)
draw.rectangle([(0, 0), (w, h)], fill=COLOR_BG)

# 1) Status bar (top ~90px) - neutral gray background where system icons sit
status_bar_h = 90
draw.rectangle([(0, 0), (w, status_bar_h)], fill=COLOR_STATUS)

# 2) Header / search area
# Header area sits below status bar; keep it white but give a strong blue underline
header_top = status_bar_h
header_bottom = header_top + 130  # spacious header for search field
draw.rectangle([(0, header_top), (w, header_bottom)], fill=COLOR_HEADER)

# Blue underline for the search field (indented horizontally like the app)
underline_left = 48
underline_right = w - 48
underline_y = header_bottom - 8
underline_thickness = 6
draw.rectangle(
    [(underline_left, underline_y - underline_thickness // 2),
     (underline_right, underline_y + underline_thickness // 2)],
    fill=COLOR_PRIMARY
)

# subtle hairline divider right below the blue underline
draw.line([(0, underline_y + 12), (w, underline_y + 12)], fill=COLOR_DIV, width=1)

# 3) Main content grouping card background (rounded rectangle behind list)
# Place a subtle rounded card behind the list of "Recent" items
card_left = 24
card_right = w - 24
card_top = header_bottom + 70
card_bottom = 1760
card_radius = 14
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                       radius=card_radius, fill=COLOR_CARD, outline=None)

# Add a faint inner top divider to separate heading from list (no text drawn)
draw.line([(card_left + 8, card_top + 68), (card_right - 8, card_top + 68)],
          fill=COLOR_DIV, width=1)

# 4) Row separators for the list items (do not draw icons/text)
# Use the observed vertical spacing from the screenshot to place separators.
# We'll draw separators across the card interior with left padding.
row_left = card_left + 24
row_right = card_right - 24
# Y positions approximating the center/top of each row group (based on screenshot)
row_starts = [534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
for y in row_starts:
    # Only draw separators that fall within the card bounds
    if card_top + 10 < y < card_bottom - 10:
        draw.line([(row_left, y - 10), (row_right, y - 10)], fill=COLOR_DIV, width=1)

# 5) Additional subtle vertical guide on the left (visual column for icons)
# Draw as a very faint vertical line to suggest where list icons align
guide_x = card_left + 36
draw.line([(guide_x, card_top + 20), (guide_x, card_bottom - 20)], fill=(245,245,247), width=1)

# 6) Large empty content space below the list (keeps background consistent)
content_area_top = card_bottom + 20
content_area_bottom = h - 160  # space for bottom nav
draw.rectangle([(0, content_area_top), (w, content_area_bottom)], fill=COLOR_BG)

# 7) Bottom navigation bar
nav_top = h - 156
nav_bottom = h
# top border of nav
draw.line([(0, nav_top), (w, nav_top)], fill=COLOR_NAV_BORDER, width=1)
# nav background (slightly off-white to separate from page)
draw.rectangle([(0, nav_top), (w, nav_bottom)], fill=(255,255,255))

# 8) Nav item separators (subtle) - evenly spaced vertical guides where icons will be placed
nav_cols = 5
col_w = w / nav_cols
for i in range(1, nav_cols):
    cx = int(i * col_w)
    # draw faint guide line (very subtle)
    draw.line([(cx, nav_top + 18), (cx, nav_bottom - 18)], fill=(250,250,252), width=1)

# 9) Small top shadow under header to separate it from content (soft)
shadow_y0 = header_bottom
shadow_y1 = header_bottom + 14
for i, alpha in enumerate([14, 10, 6, 3]):
    y = shadow_y0 + i * 3
    # draw progressively lighter horizontal lines
    shade = (230, 230, 235)
    draw.line([(0, y), (w, y)], fill=shade, width=1)

# Done drawing structural/background elements. Icons/text will be pasted on top later.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/00_icon_5.09.png
try:
    _c0 = get_crop(0, 60, 62)
    canvas.paste(_c0, (179, 2), _c0)
except Exception:
    pass
layout["5.09"] = [179, 2, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/01_icon_5.09.png
try:
    _c1 = get_crop(1, 60, 64)
    canvas.paste(_c1, (114, 1), _c1)
except Exception:
    pass
layout["5.09"] = [114, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/02_icon_Search_for__..png
try:
    _c2 = get_crop(2, 64, 64)
    canvas.paste(_c2, (309, 1), _c2)
except Exception:
    pass
layout["[Search_for__."] = [309, 1, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 62)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 149, 144)
    canvas.paste(_c4, (1243, 97), _c4)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 58, 62)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/06_icon_Open_Mic_Night.png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 534), _c6)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 99, 62)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/08_icon_5.09.png
try:
    _c8 = get_crop(8, 124, 108)
    canvas.paste(_c8, (53, 114), _c8)
except Exception:
    pass
layout["5.09"] = [53, 114, 177, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/09_icon_community_events.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 1398), _c9)
except Exception:
    pass
layout["community_events"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (864, 2804), _c10)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/11_icon_community_events.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 1542), _c11)
except Exception:
    pass
layout["community_events"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 822), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/13_icon_Favorites.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (576, 2804), _c13)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 534), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/15_icon_Search_for__..png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 390), _c15)
except Exception:
    pass
layout["[Search_for__."] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 678), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/17_icon_Search_for__..png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["[Search_for__."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1254), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1398), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1110), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1686), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/22_icon_Cancel.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 390), _c22)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1542), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/24_icon_community_events.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1254), _c24)
except Exception:
    pass
layout["community_events"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/25_icon_Close_current_screen.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 966), _c25)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/27_icon_Search_events.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/28_icon_Open_Mic_Night.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 678), _c28)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/29_icon_Business.png
try:
    _c29 = get_crop(29, 116, 131)
    canvas.paste(_c29, (26, 1696), _c29)
except Exception:
    pass
layout["Business"] = [26, 1696, 142, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/30_icon_Search_for__..png
try:
    _c30 = get_crop(30, 48, 65)
    canvas.paste(_c30, (383, 2), _c30)
except Exception:
    pass
layout["[Search_for__."] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/31_icon_More.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (1152, 2804), _c31)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/32_icon_Cooking.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 1110), _c32)
except Exception:
    pass
layout["Cooking"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/33_icon_Photography.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 822), _c33)
except Exception:
    pass
layout["Photography"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/34_text_5.09.png
try:
    _c34 = get_crop(34, 89, 43)
    canvas.paste(_c34, (22, 17), _c34)
except Exception:
    pass
layout["5.09"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/35_text_Recent.png
try:
    _c35 = get_crop(35, 203, 62)
    canvas.paste(_c35, (45, 299), _c35)
except Exception:
    pass
layout["Recent"] = [45, 299, 248, 361]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/36_text_Wellness.png
try:
    _c36 = get_crop(36, 177, 52)
    canvas.paste(_c36, (163, 1017), _c36)
except Exception:
    pass
layout["Wellness"] = [163, 1017, 340, 1069]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/37_text_Business.png
try:
    _c37 = get_crop(37, 173, 43)
    canvas.paste(_c37, (165, 1740), _c37)
except Exception:
    pass
layout["Business"] = [165, 1740, 338, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/38_clickable_Wellness.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 966), _c38)
except Exception:
    pass
layout["Wellness"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_02_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-4/39_clickable_Business.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Business"] = [48, 1686, 1392, 1830]
