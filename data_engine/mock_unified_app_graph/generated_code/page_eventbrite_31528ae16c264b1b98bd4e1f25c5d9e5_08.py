# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_08
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10.png
# step_index: 8/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background fills, status bar, header/toolbars, section card backgrounds, and separators.

# Colors
status_bar_color = "#bdbdbd"        # light grey status bar
toolbar_color = "#ffffff"           # white toolbar
divider_color = "#e6e6e6"           # subtle divider
chip_band_color = "#f6fbff"         # very light blue band behind filter chips
page_bg = "#ffffff"                 # page background (canvas already white)
card_shadow = "#f0f0f3"             # soft shadow for cards
card_bg = "#ffffff"                 # card background
image_placeholder_bg = "#eef3f7"    # light background for image containers
bottom_bar_bg = "#ffffff"           # bottom navigation background
bottom_bar_border = "#e9e9ea"       # top border for bottom nav

# Canvas is already white; if needed, fill entire canvas with page background
draw.rectangle([(0, 0), (1440, 2960)], fill=page_bg)

# Status bar (top phone bar)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill=status_bar_color)

# Top toolbar / search header area
toolbar_top = status_h
toolbar_bottom = 160
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill=toolbar_color)
# subtle bottom divider under toolbar
draw.line([(48, toolbar_bottom), (1440-48, toolbar_bottom)], fill=divider_color, width=2)

# Light band behind filter chips / location row
chip_band_top = 240
chip_band_bottom = 420
draw.rectangle([(0, chip_band_top), (1440, chip_band_bottom)], fill=chip_band_color)
# add a subtle horizontal divider below chips
draw.line([(48, chip_band_bottom + 6), (1440-48, chip_band_bottom + 6)], fill=divider_color, width=1)

# Main content area separators
# thin divider under the main header/search area (a bit lower than toolbar divider)
draw.line([(48, 220), (1440-48, 220)], fill=divider_color, width=1)

# First event card group background (subtle shadow and white rounded card)
# Based on detected group at (48, 953) size (1344x1108)
card1_x0 = 36
card1_y0 = 920
card1_x1 = 1404
card1_y1 = 2090
card_radius = 28
# shadow
draw.rounded_rectangle(
    [(card1_x0 + 8, card1_y0 + 8), (card1_x1 + 8, card1_y1 + 8)],
    radius=card_radius + 2,
    fill=card_shadow
)
# white card background
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=card_radius,
    fill=card_bg,
    outline=None
)
# subtle inner divider separating title area from image area inside the card
# approximate position: draw a light line about 160px below the top of the card
inner_div_y = card1_y0 + 160
draw.line([(card1_x0 + 24, inner_div_y), (card1_x1 - 24, inner_div_y)], fill=divider_color, width=1)

# Image placeholder background inside first card (to sit behind the image that will be pasted)
# We'll draw a rounded rectangle where the event image area is expected within this card.
# Place it starting slightly below the inner divider with margins
img1_x0 = 48
img1_y0 = inner_div_y + 24
img1_x1 = 1392
# approximate height for first image region inside the card
img1_y1 = img1_y0 + 640
draw.rounded_rectangle([(img1_x0, img1_y0), (img1_x1, img1_y1)], radius=18, fill=image_placeholder_bg)

# Separator line between first card and the next list item
sep_y = card1_y1 + 22
draw.line([(48, sep_y), (1440-48, sep_y)], fill=divider_color, width=1)

# Second event image/group background (shadow + container)
# Detected event image at (48, 2109) size (1344x707)
img2_x0 = 36
img2_y0 = 2100
img2_x1 = 1404
img2_y1 = 2880  # slightly extended to allow for rounded corner shadow
img2_radius = 22
# shadow for second image block
draw.rounded_rectangle(
    [(img2_x0 + 6, img2_y0 + 6), (img2_x1 + 6, img2_y1 + 6)],
    radius=img2_radius + 2,
    fill=card_shadow
)
# container (light background) where the image will be pasted on top
# Use slightly tighter bounds matching the detected image area (48,2109,1344x707)
img2_container = (48, 2109, 48 + 1344, 2109 + 707)
draw.rounded_rectangle([(img2_container[0], img2_container[1]), (img2_container[2], img2_container[3])],
                       radius=18, fill=image_placeholder_bg)

# Small label band behind the "Free" badge locations (subtle rounded rectangle background for badges)
# There are small badges near the top of each event; create faint rounded rectangles where badges appear.
badge1_box = (48, 1000, 48 + 90, 1000 + 42)
badge2_box = (48, 2115, 48 + 90, 2115 + 42)
draw.rounded_rectangle([badge1_box[0], badge1_box[1], badge1_box[2], badge1_box[3]], radius=8, fill="#eef6ef")
draw.rounded_rectangle([badge2_box[0], badge2_box[1], badge2_box[2], badge2_box[3]], radius=8, fill="#eef6ef")

# Bottom navigation bar background and top border
bottom_bar_top = 2804
bottom_bar_bottom = 2960
draw.rectangle([(0, bottom_bar_top), (1440, bottom_bar_bottom)], fill=bottom_bar_bg)
draw.line([(0, bottom_bar_top), (1440, bottom_bar_top)], fill=bottom_bar_border, width=2)

# Subtle shadow above bottom nav to separate content and nav
draw.rectangle([(0, bottom_bar_top-6), (1440, bottom_bar_top)], fill=card_shadow)

# Final subtle left margin guide (vertical) for alignment (very faint)
draw.line([(48, toolbar_bottom + 4), (48, bottom_bar_top - 10)], fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/00_icon_Business.png
try:
    _c0 = get_crop(0, 252, 112)
    canvas.paste(_c0, (1041, 405), _c0)
except Exception:
    pass
layout["Business"] = [1041, 405, 1293, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/01_icon_Music.png
try:
    _c1 = get_crop(1, 199, 113)
    canvas.paste(_c1, (842, 405), _c1)
except Exception:
    pass
layout["Music"] = [842, 405, 1041, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 1344, 380)
    canvas.paste(_c2, (48, 525), _c2)
except Exception:
    pass
layout["Anytime"] = [48, 525, 1392, 905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/03_icon_Fo.png
try:
    _c3 = get_crop(3, 140, 110)
    canvas.paste(_c3, (1295, 406), _c3)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1435, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/04_icon_1_Filter.png
try:
    _c4 = get_crop(4, 536, 144)
    canvas.paste(_c4, (0, 259), _c4)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1469), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1469, 1236, 1613]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2625), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2625, 1236, 2769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1469), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1469, 1380, 1613]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2625), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2625, 1380, 2769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/09_icon_Fo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Fo("] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/10_icon_7.55.png
try:
    _c10 = get_crop(10, 126, 114)
    canvas.paste(_c10, (53, 114), _c10)
except Exception:
    pass
layout["7.55"] = [53, 114, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/11_icon_Fitness.png
try:
    _c11 = get_crop(11, 70, 65)
    canvas.paste(_c11, (307, 0), _c11)
except Exception:
    pass
layout["Fitness"] = [307, 0, 377, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/12_icon_7.55.png
try:
    _c12 = get_crop(12, 62, 64)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["7.55"] = [180, 0, 242, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 106, 61)
    canvas.paste(_c13, (1204, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1204, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/14_icon_HIIT_Bodyweight_Pilates_Weekly_Fitness.png
try:
    _c14 = get_crop(14, 1344, 1108)
    canvas.paste(_c14, (48, 953), _c14)
except Exception:
    pass
layout["HIIT_Bodyweight_+_Pilates"] = [48, 953, 1392, 2061]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/15_icon_Fitness.png
try:
    _c15 = get_crop(15, 52, 64)
    canvas.paste(_c15, (249, 0), _c15)
except Exception:
    pass
layout["Fitness"] = [249, 0, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/16_icon_7.55.png
try:
    _c16 = get_crop(16, 62, 65)
    canvas.paste(_c16, (114, 0), _c16)
except Exception:
    pass
layout["7.55"] = [114, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/17_icon_Fitness.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Fitness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/18_icon_KNSER_PERMANEMIEL.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (576, 2804), _c18)
except Exception:
    pass
layout["KNSER_PERMANEMIEL"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 65, 60)
    canvas.paste(_c19, (1317, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1317, 0, 1382, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/20_icon_San_Francisco.png
try:
    _c20 = get_crop(20, 536, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 49, 60)
    canvas.paste(_c21, (385, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [385, 3, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/22_icon_Search_events.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/23_icon_More.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/24_icon_Promoted.png
try:
    _c24 = get_crop(24, 244, 59)
    canvas.paste(_c24, (83, 801), _c24)
except Exception:
    pass
layout["Promoted"] = [83, 801, 327, 860]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/25_icon_Home.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/26_icon_Tickets.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/27_icon_Tickets.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/28_icon_HIIT_Bodyweight_Pilates_Weekly_Fitness.png
try:
    _c28 = get_crop(28, 1344, 1108)
    canvas.paste(_c28, (48, 953), _c28)
except Exception:
    pass
layout["HIIT_Bodyweight_+_Pilates"] = [48, 953, 1392, 2061]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/29_text_7.55.png
try:
    _c29 = get_crop(29, 92, 43)
    canvas.paste(_c29, (22, 17), _c29)
except Exception:
    pass
layout["7.55"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/30_text_Free.png
try:
    _c30 = get_crop(30, 77, 38)
    canvas.paste(_c30, (117, 524), _c30)
except Exception:
    pass
layout["Free"] = [117, 524, 194, 562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/31_text_Empowering_Wisdom_Parenting_Circle.png
try:
    _c31 = get_crop(31, 1344, 380)
    canvas.paste(_c31, (48, 525), _c31)
except Exception:
    pass
layout["Empowering_Wisdom_Parenti"] = [48, 525, 1392, 905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/32_text_Tue_Apr_30.png
try:
    _c32 = get_crop(32, 230, 54)
    canvas.paste(_c32, (90, 673), _c32)
except Exception:
    pass
layout["Tue,_Apr_30"] = [90, 673, 320, 727]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/33_text_8.15_PM_EDT.png
try:
    _c33 = get_crop(33, 251, 45)
    canvas.paste(_c33, (339, 674), _c33)
except Exception:
    pass
layout["8.15_PM_EDT"] = [339, 674, 590, 719]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/34_text_Online.png
try:
    _c34 = get_crop(34, 129, 45)
    canvas.paste(_c34, (91, 741), _c34)
except Exception:
    pass
layout["Online"] = [91, 741, 220, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/35_text_WEEKLY.png
try:
    _c35 = get_crop(35, 311, 77)
    canvas.paste(_c35, (81, 983), _c35)
except Exception:
    pass
layout["WEEKLY"] = [81, 983, 392, 1060]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/36_text_Sun_May_5.png
try:
    _c36 = get_crop(36, 228, 57)
    canvas.paste(_c36, (88, 1894), _c36)
except Exception:
    pass
layout["Sun,_May_5"] = [88, 1894, 316, 1951]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/37_text_1O_00_AM_PDT.png
try:
    _c37 = get_crop(37, 279, 48)
    canvas.paste(_c37, (335, 1896), _c37)
except Exception:
    pass
layout["1O:00_AM_PDT"] = [335, 1896, 614, 1944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/38_text_Thrive_City.png
try:
    _c38 = get_crop(38, 207, 57)
    canvas.paste(_c38, (93, 1960), _c38)
except Exception:
    pass
layout["Thrive_City"] = [93, 1960, 300, 2017]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/39_text_WEEKLY.png
try:
    _c39 = get_crop(39, 308, 79)
    canvas.paste(_c39, (84, 2137), _c39)
except Exception:
    pass
layout["WEEKLY"] = [84, 2137, 392, 2216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/40_text_CLASSES.png
try:
    _c40 = get_crop(40, 352, 79)
    canvas.paste(_c40, (79, 2281), _c40)
except Exception:
    pass
layout["CLASSES"] = [79, 2281, 431, 2360]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/41_text_PRESENTED_By.png
try:
    _c41 = get_crop(41, 164, 30)
    canvas.paste(_c41, (90, 2370), _c41)
except Exception:
    pass
layout["PRESENTED_By"] = [90, 2370, 254, 2400]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/42_text_KNSER_PERMANEMIEL.png
try:
    _c42 = get_crop(42, 261, 39)
    canvas.paste(_c42, (132, 2412), _c42)
except Exception:
    pass
layout["KNSER_PERMANEMIEL"] = [132, 2412, 393, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_08_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-10/43_clickable_Event_s_image.png
try:
    _c43 = get_crop(43, 1344, 707)
    canvas.paste(_c43, (48, 2109), _c43)
except Exception:
    pass
layout["Event's_image"] = [48, 2109, 1392, 2816]
