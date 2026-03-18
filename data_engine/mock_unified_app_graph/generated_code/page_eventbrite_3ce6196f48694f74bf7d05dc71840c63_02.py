# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_02
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4.png
# step_index: 2/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile UI page.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (189, 189, 189)      # light gray for status bar
accent_blue = (30, 86, 214)             # search underline / accent
card_bg = (247, 248, 250)               # very light card background
section_divider = (224, 224, 228)       # subtle divider
separator_color = (238, 238, 240)       # thin separators between list items
bottom_border = (225, 226, 230)         # top border of bottom nav
shadow_strip = (245, 246, 248)          # subtle shadow areas

w, h = canvas.size

# 1) Status bar area (top)
status_bar_height = 84
draw.rectangle([(0, 0), (w, status_bar_height)], fill=status_bar_color)

# 2) Search/header area background (keeps white but we add structure lines)
search_top = status_bar_height
search_bottom = 160
# white background (canvas already white) but add subtle shadow strip under status bar
draw.rectangle([(0, search_top), (w, search_bottom)], fill=(255, 255, 255))
# Blue underline for the search field (thin accent line)
underline_y = 140
draw.line([(48, underline_y), (w - 48, underline_y)], fill=accent_blue, width=6)
# subtle divider below search underline
draw.line([(48, underline_y + 12), (w - 48, underline_y + 12)], fill=section_divider, width=1)

# 3) "Recent" area header divider (structural)
recent_div_top = 240
recent_div_bottom = 300
# light background strip behind the header area to separate search and list
draw.rectangle([(24, recent_div_top), (w - 24, recent_div_bottom)], fill=(255, 255, 255))
# faint horizontal rule underneath header area
draw.line([(24, recent_div_bottom + 4), (w - 24, recent_div_bottom + 4)], fill=section_divider, width=1)

# 4) Rounded card background for the list of recent items
list_card_x0 = 32
list_card_x1 = w - 32
list_card_y0 = recent_div_bottom + 8
list_card_y1 = 1888
card_radius = 14
draw.rounded_rectangle([(list_card_x0, list_card_y0), (list_card_x1, list_card_y1)], radius=card_radius, fill=card_bg)

# 5) Separators between list items (drawn across the card)
# Detected clickable list item top positions (from detection): 534,678,822,966,1110,1254,1398,1542,1686
# We'll draw separators at the boundaries between them (the top y positions are the separators)
separator_positions = [678, 822, 966, 1110, 1254, 1398, 1542, 1686, 1830]
for y in separator_positions:
    # Draw a subtle line across the content area within card margins
    draw.line([(48, y), (w - 48, y)], fill=separator_color, width=1)

# 6) Subtle shadow/highlight inside the card top to give depth
draw.line([(list_card_x0 + 6, list_card_y0 + 2), (list_card_x1 - 6, list_card_y0 + 2)], fill=shadow_strip, width=1)

# 7) Bottom navigation bar top border and subtle background separation
bottom_nav_top = 2804
draw.line([(0, bottom_nav_top), (w, bottom_nav_top)], fill=bottom_border, width=2)
# Slightly lighter band above the border to emulate the subtle divider/shadow
draw.line([(0, bottom_nav_top - 2), (w, bottom_nav_top - 2)], fill=shadow_strip, width=1)

# 8) Additional structural vertical guides (subtle) to suggest content margins
left_guide_x = 48
right_guide_x = w - 48
draw.line([(left_guide_x, list_card_y0), (left_guide_x, list_card_y1)], fill=(255, 255, 255), width=0)  # no-op but preserves margin intent
draw.line([(right_guide_x, list_card_y0), (right_guide_x, list_card_y1)], fill=(255, 255, 255), width=0)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/00_icon_7.24.png
try:
    _c0 = get_crop(0, 58, 62)
    canvas.paste(_c0, (181, 1), _c0)
except Exception:
    pass
layout["7.24"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/01_icon_7.24.png
try:
    _c1 = get_crop(1, 59, 63)
    canvas.paste(_c1, (114, 1), _c1)
except Exception:
    pass
layout["7.24"] = [114, 1, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/02_icon_Search_for_-..png
try:
    _c2 = get_crop(2, 64, 64)
    canvas.paste(_c2, (309, 1), _c2)
except Exception:
    pass
layout["(Search_for:-."] = [309, 1, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 62)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 149, 144)
    canvas.paste(_c4, (1243, 97), _c4)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 56, 63)
    canvas.paste(_c5, (1317, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 99, 62)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/07_icon_Music_Festival.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 534), _c7)
except Exception:
    pass
layout["Music_Festival"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/08_icon_7.24.png
try:
    _c8 = get_crop(8, 123, 109)
    canvas.paste(_c8, (54, 114), _c8)
except Exception:
    pass
layout["7.24"] = [54, 114, 177, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/09_icon_7.24.png
try:
    _c9 = get_crop(9, 96, 62)
    canvas.paste(_c9, (13, 1), _c9)
except Exception:
    pass
layout["7.24"] = [13, 1, 109, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/10_icon_Science_Tech.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 1398), _c10)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/11_icon_Favorites.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (576, 2804), _c11)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 822), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 534), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 678), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 1254), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/16_icon_Search_for_-..png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["(Search_for:-."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1398), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1110), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/19_icon_Tickets.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (864, 2804), _c19)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 390), _c20)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1542), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 1686), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/23_icon_Music_Festival.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 390), _c23)
except Exception:
    pass
layout["Music_Festival"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/24_icon_Home.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/25_icon_Basketball.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 1542), _c25)
except Exception:
    pass
layout["Basketball"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/26_icon_Search_for_-..png
try:
    _c26 = get_crop(26, 48, 65)
    canvas.paste(_c26, (383, 2), _c26)
except Exception:
    pass
layout["(Search_for:-."] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/27_icon_Science_Tech.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1254), _c27)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/28_icon_Close_current_screen.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (1248, 966), _c28)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/29_icon_Search_events.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (288, 2804), _c29)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/30_icon_Education.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 678), _c30)
except Exception:
    pass
layout["Education"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/31_icon_More.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (1152, 2804), _c31)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/32_icon_Food_Drink.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 1110), _c32)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/33_icon_Exhibition.png
try:
    _c33 = get_crop(33, 115, 128)
    canvas.paste(_c33, (26, 1697), _c33)
except Exception:
    pass
layout["Exhibition"] = [26, 1697, 141, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/34_text_Recent.png
try:
    _c34 = get_crop(34, 203, 62)
    canvas.paste(_c34, (45, 299), _c34)
except Exception:
    pass
layout["Recent"] = [45, 299, 248, 361]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/35_text_Education.png
try:
    _c35 = get_crop(35, 195, 50)
    canvas.paste(_c35, (162, 872), _c35)
except Exception:
    pass
layout["Education"] = [162, 872, 357, 922]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/36_text_Music.png
try:
    _c36 = get_crop(36, 124, 53)
    canvas.paste(_c36, (163, 1014), _c36)
except Exception:
    pass
layout["Music"] = [163, 1014, 287, 1067]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/37_text_Exhibition.png
try:
    _c37 = get_crop(37, 191, 49)
    canvas.paste(_c37, (164, 1735), _c37)
except Exception:
    pass
layout["Exhibition"] = [164, 1735, 355, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/38_clickable_Education.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 822), _c38)
except Exception:
    pass
layout["Education"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/39_clickable_Music.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 966), _c39)
except Exception:
    pass
layout["Music"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_02_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-4/40_clickable_Exhibition.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 1686), _c40)
except Exception:
    pass
layout["Exhibition"] = [48, 1686, 1392, 1830]
