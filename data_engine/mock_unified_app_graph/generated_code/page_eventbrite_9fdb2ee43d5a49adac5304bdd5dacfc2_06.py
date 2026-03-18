# page_id: page_eventbrite_9fdb2ee43d5a49adac5304bdd5dacfc2_06
# screenshot: 2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8.png
# step_index: 6/8
# task: Open Eventbrite. Look up 'Pet' events. Filter by events happening this weekend. Select the third non-promoted event from the results - how much are the tickets for the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI layout for Event list screen
# Uses given 'canvas' (1440x2960 RGB) and 'draw' (ImageDraw)
# Fonts available: font_sm, font_md, font_lg, font_xl

# Helper to convert hex
def hc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# Colors
bg_color = hc("#fbfcfd")        # overall page background
status_bar_color = hc("#cfcfd1")# status bar slightly darker
header_bg = hc("#ffffff")       # header/toolbar background
divider = hc("#e6e6e9")         # subtle dividers
card_bg = hc("#ffffff")         # card container
image_placeholder = hc("#eef3f6")  # image/content area background
muted_shadow = hc("#f1f3f5")    # soft shadow/edge
bottom_nav_bg = hc("#ffffff")   # bottom nav background

# Fill full background
draw.rectangle([(0,0),(1440,2960)], fill=bg_color)

# Status bar area (approx height 96px)
status_h = 96
draw.rectangle([(0,0),(1440,status_h)], fill=status_bar_color)

# Header / toolbar (below status bar)
header_top = status_h
header_bottom = 192
draw.rectangle([(0,header_top),(1440,header_bottom)], fill=header_bg)
# Header bottom divider
draw.line([(48, header_bottom),(1392, header_bottom)], fill=divider, width=2)

# Separator under filters / above list (place under detected chips area)
filters_bottom_sep_y = 520
draw.line([(48, filters_bottom_sep_y),(1392, filters_bottom_sep_y)], fill=divider, width=1)

# First event card container (white rounded card)
card1_left = 32
card1_top = 600
card1_right = 1408
card1_bottom = 1768
draw.rounded_rectangle(
    [(card1_left, card1_top), (card1_right, card1_bottom)],
    radius=28, fill=card_bg, outline=muted_shadow, width=1
)
# Subtle shadow under first card
draw.rectangle([(card1_left+6, card1_bottom+4), (card1_right-6, card1_bottom+8)], fill=muted_shadow)

# Image/content placeholder inside first card (do not draw any text/icons)
img1_left = 48
img1_top = 676
img1_right = 1392
img1_bottom = 1724
draw.rounded_rectangle(
    [(img1_left, img1_top), (img1_right, img1_bottom)],
    radius=20, fill=image_placeholder, outline=hc("#e2e6ea"), width=1
)

# Thin separator between cards
sep_y = card1_bottom + 8
draw.line([(48, sep_y),(1392, sep_y)], fill=divider, width=1)

# Second event card container (white rounded card)
card2_left = 32
card2_top = 1740
card2_right = 1408
card2_bottom = 2836
draw.rounded_rectangle(
    [(card2_left, card2_top), (card2_right, card2_bottom)],
    radius=28, fill=card_bg, outline=muted_shadow, width=1
)
# Subtle shadow under second card
draw.rectangle([(card2_left+6, card2_bottom+4), (card2_right-6, card2_bottom+8)], fill=muted_shadow)

# Image/content placeholder inside second card
img2_left = 48
img2_top = 1772
img2_right = 1392
img2_bottom = 2816
draw.rounded_rectangle(
    [(img2_left, img2_top), (img2_right, img2_bottom)],
    radius=20, fill=image_placeholder, outline=hc("#e2e6ea"), width=1
)

# Bottom navigation bar background and top divider
bottom_nav_top = 2804
draw.rectangle([(0, bottom_nav_top), (1440, 2960)], fill=bottom_nav_bg)
draw.line([(0, bottom_nav_top),(1440, bottom_nav_top)], fill=divider, width=2)

# Additional subtle separators for visual grouping (do not draw any icons/text)
# Separator under header search area
draw.line([(48, 192+6),(1392, 192+6)], fill=hc("#f3f4f6"), width=1)

# Slight left margin guide (accent) - very subtle vertical line
draw.line([(48, 200),(48, 2760)], fill=hc("#fbfbfc"), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (954, 410), _c0)
except Exception:
    pass
layout["Music"] = [954, 410, 1141, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/01_icon_This_Weekend.png
try:
    _c1 = get_crop(1, 504, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["This_Weekend"] = [438, 410, 942, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/02_icon_Business.png
try:
    _c2 = get_crop(2, 239, 103)
    canvas.paste(_c2, (1153, 410), _c2)
except Exception:
    pass
layout["Business"] = [1153, 410, 1392, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2288), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2288, 1236, 2432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2288), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2288, 1380, 2432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/07_icon_April_Community_Tours_Meet_the_Animals.png
try:
    _c7 = get_crop(7, 1344, 1044)
    canvas.paste(_c7, (48, 1772), _c7)
except Exception:
    pass
layout["April_Community_Tours!_Me"] = [48, 1772, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/08_icon_Close_current_screen.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/09_icon_4.48.png
try:
    _c9 = get_crop(9, 55, 65)
    canvas.paste(_c9, (116, 1), _c9)
except Exception:
    pass
layout["4.48"] = [116, 1, 171, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/10_icon_4.48.png
try:
    _c10 = get_crop(10, 54, 63)
    canvas.paste(_c10, (183, 1), _c10)
except Exception:
    pass
layout["4.48"] = [183, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/11_icon_Pets.png
try:
    _c11 = get_crop(11, 62, 62)
    canvas.paste(_c11, (310, 1), _c11)
except Exception:
    pass
layout["Pets"] = [310, 1, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/12_icon_GROW.png
try:
    _c12 = get_crop(12, 1344, 1048)
    canvas.paste(_c12, (48, 676), _c12)
except Exception:
    pass
layout["GROW"] = [48, 676, 1392, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/13_icon_4.48.png
try:
    _c13 = get_crop(13, 117, 110)
    canvas.paste(_c13, (59, 117), _c13)
except Exception:
    pass
layout["4.48"] = [59, 117, 176, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/14_icon_Washington.png
try:
    _c14 = get_crop(14, 493, 144)
    canvas.paste(_c14, (0, 259), _c14)
except Exception:
    pass
layout["Washington"] = [0, 259, 493, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1236, 1192), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/16_icon_April_Community_Tours_Meet_the_Animals.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (576, 2804), _c16)
except Exception:
    pass
layout["April_Community_Tours!_Me"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/17_icon_Pets.png
try:
    _c17 = get_crop(17, 47, 64)
    canvas.paste(_c17, (250, 0), _c17)
except Exception:
    pass
layout["Pets"] = [250, 0, 297, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 96, 62)
    canvas.paste(_c18, (1210, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1210, 0, 1306, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/19_icon_Pets.png
try:
    _c19 = get_crop(19, 203, 100)
    canvas.paste(_c19, (177, 121), _c19)
except Exception:
    pass
layout["Pets"] = [177, 121, 380, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 55, 62)
    canvas.paste(_c20, (1318, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1318, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 74, 81)
    canvas.paste(_c21, (19, 1196), _c21)
except Exception:
    pass
layout["icon_21"] = [19, 1196, 93, 1277]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/22_icon_4.48.png
try:
    _c22 = get_crop(22, 93, 63)
    canvas.paste(_c22, (14, 1), _c22)
except Exception:
    pass
layout["4.48"] = [14, 1, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/23_icon_Rosie_s_Farm_Sanctuary.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["Rosie's_Farm_Sanctuary"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 47, 62)
    canvas.paste(_c24, (384, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [384, 2, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/25_icon_April_Community_Tours_Meet_the_Animals.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["April_Community_Tours!_Me"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/26_icon_April_Community_Tours_Meet_the_Animals.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["April_Community_Tours!_Me"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/27_text_28_events.png
try:
    _c27 = get_crop(27, 372, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["28_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/28_clickable_Pets.png
try:
    _c28 = get_crop(28, 1344, 191)
    canvas.paste(_c28, (48, 72), _c28)
except Exception:
    pass
layout["Pets"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_06_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-8/29_clickable_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
