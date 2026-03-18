# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_07
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9.png
# step_index: 7/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FAFAFB")

# Status bar (top ~56px) - muted gray like in the screenshot
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#BDBDBD")

# Header / toolbar area below status bar
header_h = 140
header_top = status_h
header_bottom = status_h + header_h
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# Header bottom divider
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#E4E4E6", width=2)

# Light search divider line a little lower (matches subtle UI separators)
search_div_y = header_bottom + 40
draw.line([(48, search_div_y), (1392, search_div_y)], fill="#F0F0F2", width=1)

# Filter/pills background strip (subtle pale-blue area behind chips)
chips_top = 320
chips_bottom = 480
draw.rectangle([(0, chips_top), (1440, chips_bottom)], fill="#F5FAFF")

# A faint horizontal rule under the chips area
draw.line([(48, chips_bottom + 4), (1392, chips_bottom + 4)], fill="#ECEEF0", width=1)

# First event card background (rounded rectangle + subtle shadow)
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1175
shadow_offset = 10
# shadow
draw.rounded_rectangle(
    [(card1_x + shadow_offset, card1_y + shadow_offset),
     (card1_x + card1_w + shadow_offset, card1_y + card1_h + shadow_offset)],
    radius=22, fill="#ECECEC"
)
# card background
draw.rounded_rectangle(
    [(card1_x, card1_y), (card1_x + card1_w, card1_y + card1_h)],
    radius=20, fill="#FFFFFF", outline="#E6E6E8", width=1
)

# Thin separator line under first card (section separation)
sep_y = card1_y + card1_h + 24
draw.line([(48, sep_y), (1392, sep_y)], fill="#F0F0F2", width=1)

# Second event card background (rounded rectangle + subtle shadow)
card2_x, card2_y = 48, 1899
card2_w, card2_h = 1344, 917
# shadow
draw.rounded_rectangle(
    [(card2_x + shadow_offset, card2_y + shadow_offset),
     (card2_x + card2_w + shadow_offset, card2_y + card2_h + shadow_offset)],
    radius=22, fill="#ECECEC"
)
# card background
draw.rounded_rectangle(
    [(card2_x, card2_y), (card2_x + card2_w, card2_y + card2_h)],
    radius=20, fill="#FFFFFF", outline="#E6E6E8", width=1
)

# Divider above bottom navigation
nav_h = 108
nav_top = 2960 - nav_h
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E6E8", width=1)
# Bottom navigation background (subtle off-white)
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")

# Subtle left margin guide lines for content flow (non-intrusive, very light)
draw.line([(48, header_bottom + 8), (48, 2960 - nav_h - 8)], fill="#F7F7F8", width=1)
draw.line([(1392, header_bottom + 8), (1392, 2960 - nav_h - 8)], fill="#F7F7F8", width=1)

# Small faint page baseline at very bottom (edge polish)
draw.line([(0, 2958), (1440, 2958)], fill="#EFEFF0", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/04_icon_Fo.png
try:
    _c4 = get_crop(4, 136, 111)
    canvas.paste(_c4, (1296, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1296, 406, 1432, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2415), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/08_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c8 = get_crop(8, 1344, 1175)
    canvas.paste(_c8, (48, 676), _c8)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/09_icon_Fo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Fo("] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/10_icon_4.54.png
try:
    _c10 = get_crop(10, 122, 112)
    canvas.paste(_c10, (56, 114), _c10)
except Exception:
    pass
layout["4.54"] = [56, 114, 178, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/11_icon_48_0652.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1092, 2415), _c11)
except Exception:
    pass
layout["48+0652"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 66, 62)
    canvas.paste(_c12, (308, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [308, 1, 374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/13_icon_Open_Mic_Night.png
try:
    _c13 = get_crop(13, 1344, 191)
    canvas.paste(_c13, (48, 72), _c13)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 103, 62)
    canvas.paste(_c14, (1207, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1207, 0, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/15_icon_4.54.png
try:
    _c15 = get_crop(15, 58, 64)
    canvas.paste(_c15, (182, 0), _c15)
except Exception:
    pass
layout["4.54"] = [182, 0, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 63)
    canvas.paste(_c16, (247, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [247, 1, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/17_icon_4.54.png
try:
    _c17 = get_crop(17, 60, 65)
    canvas.paste(_c17, (114, 0), _c17)
except Exception:
    pass
layout["4.54"] = [114, 0, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 61, 62)
    canvas.paste(_c18, (1317, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1317, 0, 1378, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/19_icon_Shortcake_and_the_Teletubbiesl.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Shortcake_and_the_Teletub"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/20_icon_Leading_Role_Store_Opening_with_Strawber.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Leading_Role_Store_Openin"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/21_icon_Open_Mic_Night.png
try:
    _c21 = get_crop(21, 47, 62)
    canvas.paste(_c21, (384, 2), _c21)
except Exception:
    pass
layout["Open_Mic_Night"] = [384, 2, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/22_icon_Leading_Role_Store_Opening_with_Strawber.png
try:
    _c22 = get_crop(22, 1344, 917)
    canvas.paste(_c22, (48, 1899), _c22)
except Exception:
    pass
layout["Leading_Role_Store_Openin"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/23_icon_Los_Angeles.png
try:
    _c23 = get_crop(23, 492, 144)
    canvas.paste(_c23, (0, 259), _c23)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/24_icon_4.54.png
try:
    _c24 = get_crop(24, 104, 63)
    canvas.paste(_c24, (9, 0), _c24)
except Exception:
    pass
layout["4.54"] = [9, 0, 113, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/25_icon_Leading_Role_Store_Opening_with_Strawber.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Leading_Role_Store_Openin"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/26_icon_Free.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Free"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/27_icon_Promoted.png
try:
    _c27 = get_crop(27, 242, 66)
    canvas.paste(_c27, (84, 1744), _c27)
except Exception:
    pass
layout["Promoted"] = [84, 1744, 326, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/29_text_711_events.png
try:
    _c29 = get_crop(29, 372, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["711_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/30_text_313_N_Beverly_Dr.png
try:
    _c30 = get_crop(30, 323, 55)
    canvas.paste(_c30, (90, 1686), _c30)
except Exception:
    pass
layout["313_N_Beverly_Dr"] = [90, 1686, 413, 1741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_07_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-9/31_text_GRALD.png
try:
    _c31 = get_crop(31, 444, 97)
    canvas.paste(_c31, (486, 1885), _c31)
except Exception:
    pass
layout["GRALD"] = [486, 1885, 930, 1982]
