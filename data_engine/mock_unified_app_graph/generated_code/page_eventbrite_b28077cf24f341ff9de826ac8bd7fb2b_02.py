# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_02
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4.png
# step_index: 2/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas.
# Uses available variables: canvas (1440x2960 PIL Image) and draw (ImageDraw)

# Colors (matched to screenshot tonal values)
bg_color = (255, 255, 255)          # main background (white)
status_bar_color = (153, 153, 153)  # top status bar grey
status_div_color = (220, 220, 220)  # subtle divider under status bar
search_underline_color = (45, 75, 230)  # strong blue underline for search
list_card_bg = (250, 250, 253)      # very subtle off-white for list background
sep_color = (235, 235, 238)         # light separators between list items
bottom_nav_top = (230, 230, 235)    # top border above bottom nav
bottom_nav_bg = (255, 255, 255)     # nav background (white)

W, H = canvas.size

# 1) Fill whole background (canvas may already be white, but set explicitly)
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# 2) Status bar area at top (~50-84px)
status_h = 84
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)
# thin divider below status bar
draw.rectangle([(0, status_h - 1), (W, status_h + 1)], fill=status_div_color)

# 3) Search area underline (blue) approx where search field sits
# Use horizontal inset matching content margins
left_inset = 48
right_inset = W - 48
underline_y_top = 136
underline_y_bot = underline_y_top + 6
draw.rectangle([(left_inset, underline_y_top), (right_inset, underline_y_bot)], fill=search_underline_color)

# 4) Subtle rounded card background behind the list of "Recent" items (keeps content readable)
list_top = 260
list_bottom = 2740  # stop above bottom nav
card_margin = 24
card_left = card_margin
card_right = W - card_margin
card_radius = 12
# Use rounded rectangle if available
try:
    draw.rounded_rectangle([(card_left, list_top), (card_right, list_bottom)], radius=card_radius, fill=list_card_bg, outline=None)
except Exception:
    # fallback to normal rectangle if rounded not available
    draw.rectangle([(card_left, list_top), (card_right, list_bottom)], fill=list_card_bg)

# 5) Separator lines between list sections (use detected Y positions as visual guides)
separators_y = [
    360,  # subtle divider under "Recent" header area
    390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686, 1840
]
for y in separators_y:
    # draw across content area (inside card margins)
    draw.line([(left_inset, y), (right_inset, y)], fill=sep_color, width=1)

# 6) Bottom navigation area background and top border
nav_height = 156
nav_top = H - nav_height
# top border line
draw.rectangle([(0, nav_top - 1), (W, nav_top)], fill=bottom_nav_top)
# nav background
draw.rectangle([(0, nav_top), (W, H)], fill=bottom_nav_bg)

# 7) Additional subtle vertical guide at left content edge (visual structure, not an icon or text)
# This provides a column alignment guide but uses a faint color so it won't conflict.
draw.line([(left_inset, status_h + 8), (left_inset, nav_top - 12)], fill=(245,245,247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/00_icon_4.44.png
try:
    _c0 = get_crop(0, 58, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["4.44"] = [180, 2, 238, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/01_icon_4.44.png
try:
    _c1 = get_crop(1, 59, 63)
    canvas.paste(_c1, (114, 2), _c1)
except Exception:
    pass
layout["4.44"] = [114, 2, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/02_icon_Search_for_-..png
try:
    _c2 = get_crop(2, 63, 62)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["(Search_for:-."] = [309, 2, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 60)
    canvas.paste(_c3, (249, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 3, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 149, 144)
    canvas.paste(_c4, (1243, 97), _c4)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 99, 62)
    canvas.paste(_c5, (1212, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 57, 62)
    canvas.paste(_c6, (1316, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/07_icon_4.44.png
try:
    _c7 = get_crop(7, 93, 63)
    canvas.paste(_c7, (15, 1), _c7)
except Exception:
    pass
layout["4.44"] = [15, 1, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/08_icon_Sports.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 534), _c8)
except Exception:
    pass
layout["Sports"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/10_icon_Search_for_-..png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["(Search_for:-."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/11_icon_4.44.png
try:
    _c11 = get_crop(11, 125, 109)
    canvas.paste(_c11, (53, 114), _c11)
except Exception:
    pass
layout["4.44"] = [53, 114, 178, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 822), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 1686), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 534), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1398), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 678), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1110), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/19_icon_Favorites.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1542), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 966), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 390), _c23)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/24_icon_Search_events.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/25_icon_Search_for_-..png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 390), _c25)
except Exception:
    pass
layout["(Search_for:-."] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/26_icon_Art.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1254), _c26)
except Exception:
    pass
layout["Art"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/27_icon_community_events.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 678), _c27)
except Exception:
    pass
layout["community_events"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/28_icon_Yoga_session.png
try:
    _c28 = get_crop(28, 115, 130)
    canvas.paste(_c28, (26, 1697), _c28)
except Exception:
    pass
layout["Yoga_session"] = [26, 1697, 141, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/30_icon_Search_for_-..png
try:
    _c30 = get_crop(30, 47, 63)
    canvas.paste(_c30, (383, 3), _c30)
except Exception:
    pass
layout["(Search_for:-."] = [383, 3, 430, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/31_icon_community_events.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 822), _c31)
except Exception:
    pass
layout["community_events"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/32_icon_Food_and_Drink.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 1398), _c32)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/33_text_Recent.png
try:
    _c33 = get_crop(33, 203, 62)
    canvas.paste(_c33, (45, 299), _c33)
except Exception:
    pass
layout["Recent"] = [45, 299, 248, 361]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/34_text_Business.png
try:
    _c34 = get_crop(34, 179, 54)
    canvas.paste(_c34, (161, 1015), _c34)
except Exception:
    pass
layout["Business"] = [161, 1015, 340, 1069]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/35_text_Fitness.png
try:
    _c35 = get_crop(35, 140, 43)
    canvas.paste(_c35, (165, 1164), _c35)
except Exception:
    pass
layout["Fitness"] = [165, 1164, 305, 1207]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/36_text_Education.png
try:
    _c36 = get_crop(36, 195, 50)
    canvas.paste(_c36, (162, 1591), _c36)
except Exception:
    pass
layout["Education"] = [162, 1591, 357, 1641]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/37_text_Yoga_session.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 1686), _c37)
except Exception:
    pass
layout["Yoga_session"] = [48, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/38_clickable_Business.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 966), _c38)
except Exception:
    pass
layout["Business"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/39_clickable_Fitness.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1110), _c39)
except Exception:
    pass
layout["Fitness"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_02_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-4/40_clickable_Education.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 1542), _c40)
except Exception:
    pass
layout["Education"] = [48, 1542, 1392, 1686]
