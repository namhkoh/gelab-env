# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_07
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9.png
# step_index: 7/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided.
# Draw the overall background (keep dominant white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))  # light gray status bar
# subtle bottom divider for status bar
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill=(180, 180, 180), width=1)

# Header / search area background (under status bar)
header_top = status_h  # 72
header_h = 191  # matches detected large search area height
header_bottom = header_top + header_h
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Blue underline for the active search input (thin)
underline_left = 48
underline_right = 1440 - 48
underline_y = header_top + 120  # positioned within header/search area
draw.line([(underline_left, underline_y), (underline_right, underline_y)], fill=(46, 80, 255), width=4)

# Divider under the header area (subtle)
draw.line([(0, header_bottom), (1440, header_bottom)], fill=(235, 235, 235), width=1)

# Card / content grouping background (subtle rounded container behind list)
card_left = 32
card_right = 1440 - 32
card_top = header_bottom + 16  # leave a small gap after header
card_bottom = 1830  # cover the list area seen in the screenshot
card_radius = 12
# white fill with a light border to separate from the page
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                       radius=card_radius, fill=(255, 255, 255), outline=(240, 240, 240), width=1)

# Separator lines between rows inside the card
# Use the detected row top positions and heights to place separators.
separator_ys = [
    390,  # top search / recent separator
    534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686, 1830
]
sep_x1 = card_left + 16
sep_x2 = card_right - 16
for y in separator_ys:
    # only draw separators that fall within the card bounds to avoid drawing over header/footer
    if card_top < y < card_bottom:
        draw.line([(sep_x1, y), (sep_x2, y)], fill=(240, 240, 240), width=1)

# Thin left-aligned guides for list content (to hint alignment without drawing icons/text)
# These are subtle vertical rules that help structure the list but do not duplicate icons/text
guide_x = card_left + 64
draw.line([(guide_x, card_top + 8), (guide_x, card_bottom - 8)], fill=(248, 248, 248), width=1)

# Bottom navigation bar background
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(250, 250, 250))
# top border for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill=(226, 226, 226), width=1)

# Subtle "active" pill behind the center area of the nav (background only, icons will be pasted on top)
# Keep it faint and avoid duplicating icon shapes; draw a low-opacity rounded rect as highlight
active_center_x = 1440 // 2
pill_w = 140
pill_h = 88
pill_left = active_center_x - pill_w // 2
pill_right = active_center_x + pill_w // 2
pill_top = nav_top + ( (2960 - nav_top) - pill_h ) // 2
pill_bottom = pill_top + pill_h
# Very light warm tint for the selected tab background
draw.rounded_rectangle([(pill_left, pill_top), (pill_right, pill_bottom)],
                       radius=44, fill=(255, 245, 238), outline=None)

# Final subtle page bottom shadow
draw.line([(0, 2959), (1440, 2959)], fill=(240, 240, 240), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 51, 67)
    canvas.paste(_c0, (1153, 1), _c0)
except Exception:
    pass
layout["icon_0"] = [1153, 1, 1204, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/01_icon_4.50.png
try:
    _c1 = get_crop(1, 60, 64)
    canvas.paste(_c1, (114, 1), _c1)
except Exception:
    pass
layout["4.50"] = [114, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/02_icon_4.50.png
try:
    _c2 = get_crop(2, 58, 64)
    canvas.paste(_c2, (181, 0), _c2)
except Exception:
    pass
layout["4.50"] = [181, 0, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/03_icon_Search_forae.png
try:
    _c3 = get_crop(3, 63, 64)
    canvas.paste(_c3, (309, 1), _c3)
except Exception:
    pass
layout["Search_forae"] = [309, 1, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 101, 64)
    canvas.paste(_c4, (1211, 0), _c4)
except Exception:
    pass
layout["Cancel"] = [1211, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 61)
    canvas.paste(_c5, (249, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 2, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/06_icon_Wellness.png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 534), _c6)
except Exception:
    pass
layout["Wellness"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 56, 62)
    canvas.paste(_c7, (1317, 0), _c7)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/08_icon_Search_forae.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 390), _c8)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/09_icon_4.50.png
try:
    _c9 = get_crop(9, 127, 112)
    canvas.paste(_c9, (53, 115), _c9)
except Exception:
    pass
layout["4.50"] = [53, 115, 180, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/10_icon_Cancel.png
try:
    _c10 = get_crop(10, 149, 144)
    canvas.paste(_c10, (1243, 97), _c10)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/11_icon_community_events.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 1254), _c11)
except Exception:
    pass
layout["community_events"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (864, 2804), _c12)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/14_icon_Cooking.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 678), _c14)
except Exception:
    pass
layout["Cooking"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/15_icon_community_events.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1110), _c15)
except Exception:
    pass
layout["community_events"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 822), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1398), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1686), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 534), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 678), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/21_icon_community_events.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 966), _c21)
except Exception:
    pass
layout["community_events"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 1110), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1542), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/24_icon_Search_events.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/25_icon_Search_forae.png
try:
    _c25 = get_crop(25, 48, 65)
    canvas.paste(_c25, (383, 2), _c25)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/26_icon_Search_forae.png
try:
    _c26 = get_crop(26, 1344, 191)
    canvas.paste(_c26, (48, 72), _c26)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/27_icon_Favorites.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/28_icon_Home.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/29_icon_Cancel.png
try:
    _c29 = get_crop(29, 144, 144)
    canvas.paste(_c29, (1248, 390), _c29)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/30_icon_community_events.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 1398), _c30)
except Exception:
    pass
layout["community_events"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/31_icon_Close_current_screen.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (1248, 966), _c31)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/32_icon_community_events.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 822), _c32)
except Exception:
    pass
layout["community_events"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/33_icon_More.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (1152, 2804), _c33)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/34_icon_Food_and_Drink.png
try:
    _c34 = get_crop(34, 1344, 144)
    canvas.paste(_c34, (48, 1542), _c34)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/35_icon_Art.png
try:
    _c35 = get_crop(35, 115, 129)
    canvas.paste(_c35, (26, 1697), _c35)
except Exception:
    pass
layout["Art"] = [26, 1697, 141, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/36_text_4.50.png
try:
    _c36 = get_crop(36, 89, 43)
    canvas.paste(_c36, (22, 17), _c36)
except Exception:
    pass
layout["4.50"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/37_text_Recent.png
try:
    _c37 = get_crop(37, 203, 62)
    canvas.paste(_c37, (45, 299), _c37)
except Exception:
    pass
layout["Recent"] = [45, 299, 248, 361]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/38_text_Food_and_Drink.png
try:
    _c38 = get_crop(38, 286, 49)
    canvas.paste(_c38, (164, 1735), _c38)
except Exception:
    pass
layout["Food_and_Drink"] = [164, 1735, 450, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_07_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-9/39_clickable_Food_and_Drink.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 1686, 1392, 1830]
