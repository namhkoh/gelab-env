# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_04
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6.png
# step_index: 4/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
bg_color = (249, 250, 251)  # very light off-white
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar (top)
status_h = 72
status_color = (190, 190, 190)  # muted grey like Android status bar
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Header / search area
header_top = status_h
header_bottom = 180
header_color = (255, 255, 255)  # white header
draw.rectangle([(0, header_top), (canvas.width, header_bottom)], fill=header_color)
# thin divider below header
divider_color = (220, 223, 226)
draw.line([(32, header_bottom), (canvas.width - 32, header_bottom)], fill=divider_color, width=2)

# Subtle area divider for filter row (do not draw pill shapes themselves)
filters_div_y = 360
draw.line([(24, filters_div_y), (canvas.width - 24, filters_div_y)], fill=divider_color, width=1)

# Content area background (ensure a slight contrast strip where content begins)
content_top = filters_div_y + 12
draw.rectangle([(0, content_top), (canvas.width, canvas.height - 120)], fill=bg_color)

# First event card background (rounded rect, with subtle shadow)
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1115
card1_bbox = [card1_x, card1_y, card1_x + card1_w, card1_y + card1_h]

shadow_offset = 10
shadow_bbox = [card1_bbox[0] + 4, card1_bbox[1] + shadow_offset, card1_bbox[2] + 4, card1_bbox[3] + shadow_offset]
shadow_color = (235, 236, 238)
draw.rounded_rectangle(shadow_bbox, radius=28, fill=shadow_color)

card_bg = (255, 255, 255)
draw.rounded_rectangle(card1_bbox, radius=24, fill=card_bg)

# Subtle inner divider within card (for where image area meets content) - keep it light so it doesn't mimic content
# Place divider approx 40% down the card to hint image/content separation
card1_div_y = card1_y + int(card1_h * 0.48)
draw.line([(card1_x + 20, card1_div_y), (card1_x + card1_w - 20, card1_div_y)], fill=(245,245,246), width=1)

# Second event card background (rounded rect, with subtle shadow)
card2_x, card2_y = 48, 1839
card2_w, card2_h = 1344, 977
card2_bbox = [card2_x, card2_y, card2_x + card2_w, card2_y + card2_h]

shadow_bbox2 = [card2_bbox[0] + 4, card2_bbox[1] + shadow_offset, card2_bbox[2] + 4, card2_bbox[3] + shadow_offset]
draw.rounded_rectangle(shadow_bbox2, radius=28, fill=shadow_color)
draw.rounded_rectangle(card2_bbox, radius=24, fill=card_bg)

# Divider between the two cards area
between_y = card1_bbox[3] + 32
draw.line([(32, between_y), (canvas.width - 32, between_y)], fill=divider_color, width=1)

# Small promoted/label background area placeholders (no text) - subtle rounded rectangle behind where labels appear
# First label area (near top portion of card1) - light pink/peach bubble background
label1_bbox = [card1_x + 28, card1_y + int(card1_h * 0.48) - 70, card1_x + 220, card1_y + int(card1_h * 0.48) - 24]
draw.rounded_rectangle(label1_bbox, radius=18, fill=(253, 238, 240))

# Small "Free" tag background for second card (pale green) - do not draw text
label2_bbox = [card2_x + 28, card2_y + int(card2_h * 0.62), card2_x + 120, card2_y + int(card2_h * 0.62) + 40]
draw.rounded_rectangle(label2_bbox, radius=12, fill=(230, 244, 235))

# Subtle horizontal separators in content flow (between header/filter area and list)
sep_y = content_top + 40
draw.line([(24, sep_y), (canvas.width - 24, sep_y)], fill=(240,241,243), width=1)

# Bottom navigation bar background
nav_top = canvas.height - 156
nav_color = (255, 255, 255)
draw.rectangle([(0, nav_top), (canvas.width, canvas.height)], fill=nav_color)
# top border for nav
draw.line([(0, nav_top), (canvas.width, nav_top)], fill=divider_color, width=2)

# Light highlight circles where nav icons will appear (only background rings, no icons)
# Positions roughly aligned with detected clickable area (but icons themselves will be pasted)
nav_centers = [
    (72, nav_top + 78),   # left
    (360, nav_top + 78),  # search
    (648, nav_top + 78),  # center
    (936, nav_top + 78),  # saved
    (1224, nav_top + 78)  # profile
]
for cx, cy in nav_centers:
    # subtle circular background for icon hits (very light)
    draw.ellipse([(cx - 36, cy - 36), (cx + 36, cy + 36)], fill=(253, 247, 244))

# Final subtle vignette at page edges for depth
edge_shade = (245, 246, 248)
# top edge thinning
draw.rectangle([(0, header_bottom - 8), (canvas.width, header_bottom)], fill=edge_shade)
# left/right inner edges
draw.rectangle([(0, 0), (24, canvas.height)], fill=edge_shade)
draw.rectangle([(canvas.width - 24, 0), (canvas.width, canvas.height)], fill=edge_shade)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/05_icon_Foo.png
try:
    _c5 = get_crop(5, 150, 110)
    canvas.paste(_c5, (1282, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/07_icon_Wood_Fired_Master_Class.png
try:
    _c7 = get_crop(7, 1344, 1115)
    canvas.paste(_c7, (48, 676), _c7)
except Exception:
    pass
layout["Wood_Fired_Master_Class"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/08_icon_Foo.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/09_icon_7.52.png
try:
    _c9 = get_crop(9, 124, 114)
    canvas.paste(_c9, (54, 114), _c9)
except Exception:
    pass
layout["7.52"] = [54, 114, 178, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/10_icon_7.52.png
try:
    _c10 = get_crop(10, 61, 64)
    canvas.paste(_c10, (180, 0), _c10)
except Exception:
    pass
layout["7.52"] = [180, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/11_icon_Art.png
try:
    _c11 = get_crop(11, 68, 62)
    canvas.paste(_c11, (308, 1), _c11)
except Exception:
    pass
layout["Art"] = [308, 1, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 104, 61)
    canvas.paste(_c12, (1206, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1206, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/13_icon_7.52.png
try:
    _c13 = get_crop(13, 59, 65)
    canvas.paste(_c13, (115, 0), _c13)
except Exception:
    pass
layout["7.52"] = [115, 0, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/14_icon_Art.png
try:
    _c14 = get_crop(14, 55, 64)
    canvas.paste(_c14, (246, 0), _c14)
except Exception:
    pass
layout["Art"] = [246, 0, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/15_icon_Iad.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1092, 2355), _c15)
except Exception:
    pass
layout["Iad"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/16_icon_San_Francisco.png
try:
    _c16 = get_crop(16, 536, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 60, 61)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/18_icon_Enbeptenoural_strategy_Series.png
try:
    _c18 = get_crop(18, 1344, 977)
    canvas.paste(_c18, (48, 1839), _c18)
except Exception:
    pass
layout["Enbeptenoural_strategy_Se"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/19_icon_4o.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1236, 2355), _c19)
except Exception:
    pass
layout["4o"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 52, 60)
    canvas.paste(_c20, (383, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [383, 3, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/21_icon_Flavors_of_Innovation_The_Art_of_Launchi.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Flavors_of_Innovation:_Th"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/22_icon_7_7-o0.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["7_7-o0"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/23_icon_4o.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["4o"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/24_icon_Promoted.png
try:
    _c24 = get_crop(24, 244, 65)
    canvas.paste(_c24, (83, 1685), _c24)
except Exception:
    pass
layout["Promoted"] = [83, 1685, 327, 1750]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/25_icon_AarDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["^AarDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/26_text_7.52.png
try:
    _c26 = get_crop(26, 89, 43)
    canvas.paste(_c26, (22, 17), _c26)
except Exception:
    pass
layout["7.52"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/27_text_Art.png
try:
    _c27 = get_crop(27, 123, 73)
    canvas.paste(_c27, (203, 135), _c27)
except Exception:
    pass
layout["Art"] = [203, 135, 326, 208]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/28_text_4_023_events.png
try:
    _c28 = get_crop(28, 359, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["4,023_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/29_text_4.00_PM_PDT.png
try:
    _c29 = get_crop(29, 252, 45)
    canvas.paste(_c29, (336, 1560), _c29)
except Exception:
    pass
layout["4.00_PM_PDT"] = [336, 1560, 588, 1605]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/30_text_The_Forno_Piombo_Garden.png
try:
    _c30 = get_crop(30, 491, 50)
    canvas.paste(_c30, (93, 1628), _c30)
except Exception:
    pass
layout["The_Forno_Piombo_Garden"] = [93, 1628, 584, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/31_text_TL.png
try:
    _c31 = get_crop(31, 48, 21)
    canvas.paste(_c31, (98, 2787), _c31)
except Exception:
    pass
layout["TL"] = [98, 2787, 146, 2808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/32_text_n_-c.png
try:
    _c32 = get_crop(32, 127, 25)
    canvas.paste(_c32, (194, 2784), _c32)
except Exception:
    pass
layout["n_-c"] = [194, 2784, 321, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/33_text_7_7-o0.png
try:
    _c33 = get_crop(33, 115, 25)
    canvas.paste(_c33, (349, 2784), _c33)
except Exception:
    pass
layout["7_7-o0"] = [349, 2784, 464, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/34_text_AarDT.png
try:
    _c34 = get_crop(34, 145, 25)
    canvas.paste(_c34, (472, 2784), _c34)
except Exception:
    pass
layout["^AarDT"] = [472, 2784, 617, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/35_clickable_Art.png
try:
    _c35 = get_crop(35, 1344, 191)
    canvas.paste(_c35, (48, 72), _c35)
except Exception:
    pass
layout["Art"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_04_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-6/36_clickable_Home.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (0, 2804), _c36)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
