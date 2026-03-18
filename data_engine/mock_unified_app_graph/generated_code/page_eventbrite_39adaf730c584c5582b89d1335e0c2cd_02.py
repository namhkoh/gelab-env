# page_id: page_eventbrite_39adaf730c584c5582b89d1335e0c2cd_02
# screenshot: 2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4.png
# step_index: 2/6
# task: Open Eventbrite. Search for 'food and drink' events. Follow the organizer of the first event in listing.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background/structure for the UI page

# Colors
status_bar_color = (190, 190, 190)   # light gray for status bar
divider_color = (235, 236, 240)      # very light gray for separators
search_underline_color = (40, 90, 210)  # blue underline for search
card_bg = (255, 255, 255)            # white card backgrounds (subtle)
bottom_divider = (228, 229, 234)     # top border of bottom nav

w, h = canvas.size

# Status bar (top)
status_h = 72
draw.rectangle([0, 0, w, status_h], fill=status_bar_color)

# Thin shadow/divider under status bar
draw.line([(0, status_h), (w, status_h)], fill=divider_color, width=1)

# Search/header area card (behind search input)
# Use the detected search area horizontal padding (48..1392) and approximate height
search_left = 48
search_right = 48 + 1344
search_top = 72
search_height = 140
search_bottom = search_top + search_height
draw.rounded_rectangle([search_left, search_top, search_right, search_bottom],
                       radius=6, fill=card_bg, outline=None)

# Blue underline for active search (approx vertical position as in screenshot)
underline_y = search_top + 48  # put underline a bit below the top of search card
draw.line([(search_left, underline_y), (search_right, underline_y)],
          fill=search_underline_color, width=4)

# Subtle divider below the search area
draw.line([(search_left, search_bottom + 6), (search_right, search_bottom + 6)],
          fill=divider_color, width=1)

# Section separators for the list of recent items
# Use the provided clickable rows (top positions) and draw separators at their bottoms.
clickable_tops = [534, 678, 822, 966, 1110, 1254, 1542, 1686]
row_height = 144
sep_left = search_left
sep_right = search_right
for top in clickable_tops:
    y = top + row_height
    if 0 < y < h:
        draw.line([(sep_left + 8, y), (sep_right - 8, y)], fill=divider_color, width=1)

# Additional subtle separators for other stacked areas (e.g., categories)
# Some category rows appear earlier (Music Festival/Fitness/Art etc.). Add faint separators
extra_tops = [822, 966, 1110, 1254, 1542, 1686]
for top in extra_tops:
    y = top + row_height
    if 0 < y < h:
        draw.line([(sep_left, y), (sep_right, y)], fill=(245,245,247), width=1)

# Bottom navigation bar divider and subtle background
bottom_nav_top = 2804
draw.line([(0, bottom_nav_top), (w, bottom_nav_top)], fill=bottom_divider, width=2)
# Slightly darker band behind the nav icons area to define it (very subtle)
draw.rectangle([0, bottom_nav_top, w, h], fill=(255, 255, 255))

# Final light edge on left/right to match clean app margins (very subtle)
edge_color = (250, 250, 251)
draw.line([(0, 0), (0, h)], fill=edge_color, width=1)
draw.line([(w-1, 0), (w-1, h)], fill=edge_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/00_icon_7.44.png
try:
    _c0 = get_crop(0, 58, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["7.44"] = [180, 2, 238, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/01_icon_7.44.png
try:
    _c1 = get_crop(1, 59, 63)
    canvas.paste(_c1, (114, 2), _c1)
except Exception:
    pass
layout["7.44"] = [114, 2, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/02_icon_Search_for__..png
try:
    _c2 = get_crop(2, 64, 62)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["[Search_for__."] = [309, 2, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/03_icon_Cancel.png
try:
    _c3 = get_crop(3, 149, 144)
    canvas.paste(_c3, (1243, 97), _c3)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 49, 60)
    canvas.paste(_c4, (249, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 3, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 57, 63)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/06_icon_Yoga_session.png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 534), _c6)
except Exception:
    pass
layout["Yoga_session"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 99, 63)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/08_icon_7.44.png
try:
    _c8 = get_crop(8, 93, 63)
    canvas.paste(_c8, (15, 1), _c8)
except Exception:
    pass
layout["7.44"] = [15, 1, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/10_icon_7.44.png
try:
    _c10 = get_crop(10, 126, 107)
    canvas.paste(_c10, (52, 115), _c10)
except Exception:
    pass
layout["7.44"] = [52, 115, 178, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 822), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 1254), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 534), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 1398), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 678), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/16_icon_Favorites.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (576, 2804), _c16)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1110), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/18_icon_Search_for__..png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 390), _c18)
except Exception:
    pass
layout["[Search_for__."] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1686), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/20_icon_Science_Tech.png
try:
    _c20 = get_crop(20, 1344, 144)
    canvas.paste(_c20, (48, 1542), _c20)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1542), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/22_icon_Cancel.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 390), _c22)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/23_icon_Search_for__..png
try:
    _c23 = get_crop(23, 1344, 191)
    canvas.paste(_c23, (48, 72), _c23)
except Exception:
    pass
layout["[Search_for__."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/24_icon_Close_current_screen.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 966), _c24)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/25_icon_Home.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/26_icon_Food_Drink.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1398), _c26)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/27_icon_Show.png
try:
    _c27 = get_crop(27, 116, 132)
    canvas.paste(_c27, (26, 1696), _c27)
except Exception:
    pass
layout["Show"] = [26, 1696, 142, 1828]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/28_icon_Search_events.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (288, 2804), _c28)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/29_icon_Coding_Workshop.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 678), _c29)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/30_icon_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/31_icon_Search_for__..png
try:
    _c31 = get_crop(31, 47, 63)
    canvas.paste(_c31, (383, 3), _c31)
except Exception:
    pass
layout["[Search_for__."] = [383, 3, 430, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/32_text_Music_Festival.png
try:
    _c32 = get_crop(32, 274, 52)
    canvas.paste(_c32, (163, 871), _c32)
except Exception:
    pass
layout["Music_Festival"] = [163, 871, 437, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/33_text_Fitness.png
try:
    _c33 = get_crop(33, 145, 51)
    canvas.paste(_c33, (163, 1017), _c33)
except Exception:
    pass
layout["Fitness"] = [163, 1017, 308, 1068]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/34_text_Art.png
try:
    _c34 = get_crop(34, 67, 45)
    canvas.paste(_c34, (164, 1163), _c34)
except Exception:
    pass
layout["Art"] = [164, 1163, 231, 1208]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/35_text_Music.png
try:
    _c35 = get_crop(35, 122, 48)
    canvas.paste(_c35, (164, 1304), _c35)
except Exception:
    pass
layout["Music"] = [164, 1304, 286, 1352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/36_text_Show.png
try:
    _c36 = get_crop(36, 112, 43)
    canvas.paste(_c36, (163, 1740), _c36)
except Exception:
    pass
layout["Show"] = [163, 1740, 275, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/37_clickable_Music_Festival.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 822), _c37)
except Exception:
    pass
layout["Music_Festival"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/38_clickable_Fitness.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 966), _c38)
except Exception:
    pass
layout["Fitness"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/39_clickable_Art.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1110), _c39)
except Exception:
    pass
layout["Art"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/40_clickable_Music.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 1254), _c40)
except Exception:
    pass
layout["Music"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_02_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-4/41_clickable_Show.png
try:
    _c41 = get_crop(41, 1344, 144)
    canvas.paste(_c41, (48, 1686), _c41)
except Exception:
    pass
layout["Show"] = [48, 1686, 1392, 1830]
