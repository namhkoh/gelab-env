# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_02
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4.png
# step_index: 2/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (match screenshot's dominant white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~50px, muted gray)
status_h = 50
draw.rectangle([(0, 0), (1440, status_h)], fill=(196, 196, 196))

# Header / search area background (large pale surface behind search area)
search_x, search_y, search_w, search_h = 48, 72, 1344, 191
search_rect = (search_x, search_y, search_x + search_w, search_y + search_h)
draw.rounded_rectangle(search_rect, radius=18, fill=(247, 247, 250), outline=None)

# Thin divider under the search/header region
divider_y = search_y + search_h + 12
draw.line([(48, divider_y), (1392, divider_y)], fill=(220, 220, 225), width=2)

# Location row subtle background band (behind "San Francisco" area)
loc_band_h = 72
loc_band_y = 240
draw.rectangle([(0, loc_band_y), (1440, loc_band_y + loc_band_h)], fill=(255, 255, 255))

# Chips / filter pills row (rounded pill backgrounds)
chips = [
    (54, 410, 359, 103),    # Filters
    (425, 410, 400, 103),   # Anytime
    (837, 410, 187, 103),   # Music
    (1036, 410, 241, 103),  # Business
    (1282, 406, 151, 110),  # Foo
]
chip_fill = (220, 237, 250)   # pale blue chips
chip_outline = (206, 224, 236)
for (x, y, w, h) in chips:
    r = int(h / 2)
    draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=r, fill=chip_fill, outline=chip_outline, width=2)

# Subtle large heading area divider (space above event list)
heading_div_y = 540
draw.line([(48, heading_div_y), (1392, heading_div_y)], fill=(230, 230, 235), width=1)

# First event image card background (dark rounded rectangle to act as image container)
card1_x, card1_y, card1_w, card1_h = 48, 676, 1344, 1091
card1_rect = (card1_x, card1_y, card1_x + card1_w, card1_y + card1_h)
draw.rounded_rectangle(card1_rect, radius=22, fill=(24, 24, 28), outline=(18, 18, 20), width=6)

# Card body area below first image (white content background where title/metadata will be)
body1_y = card1_y + card1_h + 18
body1_h = 180
draw.rectangle([(48, body1_y), (48 + 1344, body1_y + body1_h)], fill=(255, 255, 255))
# subtle divider under the body
draw.line([(48, body1_y + body1_h + 8), (1392, body1_y + body1_h + 8)], fill=(230, 230, 235), width=1)

# Small "Promoted" tag background behind its future text (placed near first card body)
promoted_x, promoted_y, promoted_w, promoted_h = 84, 1659, 244, 67
draw.rounded_rectangle([(promoted_x, promoted_y), (promoted_x + promoted_w, promoted_y + promoted_h)],
                       radius=12, fill=(238, 246, 241), outline=(208, 230, 212))

# Second event image card background (dark rounded rectangle)
card2_x, card2_y, card2_w, card2_h = 48, 1815, 1344, 1001
card2_rect = (card2_x, card2_y, card2_x + card2_w, card2_y + card2_h)
draw.rounded_rectangle(card2_rect, radius=22, fill=(24, 24, 28), outline=(18, 18, 20), width=6)

# Card body area below second image (white content background)
body2_y = card2_y + card2_h + 18
body2_h = 220
draw.rectangle([(48, body2_y), (48 + 1344, body2_y + body2_h)], fill=(255, 255, 255))
# divider under second card
draw.line([(48, body2_y + body2_h + 8), (1392, body2_y + body2_h + 8)], fill=(230, 230, 235), width=1)

# Separator lines between major sections (thin, full-width)
sep_positions = [heading_div_y, body1_y + body1_h + 20, card2_y - 24]
for y in sep_positions:
    draw.line([(24, y), (1416, y)], fill=(240, 240, 244), width=1)

# Bottom navigation background bar
nav_h = 110
nav_top = 2960 - nav_h
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
# top hairline for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill=(220, 220, 225), width=2)

# Active center pill behind the middle nav icon (subtle colored circle), kept minimal so it doesn't duplicate icon
center_pill_center = (720, nav_top + nav_h // 2)
draw.ellipse([(center_pill_center[0] - 34, center_pill_center[1] - 34),
              (center_pill_center[0] + 34, center_pill_center[1] + 34)],
             fill=(255, 244, 235))

# Soft drop shadows under image cards (very subtle)
shadow_color = (0, 0, 0, 18)
# simulate shadows by drawing translucent rectangles if canvas supports alpha - fallback to soft gray lines
for i, y in enumerate([card1_y + card1_h, card2_y + card2_h]):
    alpha_strip_y = y + 6
    draw.rectangle([(48, alpha_strip_y), (1392, alpha_strip_y + 6)], fill=(240, 240, 245))

# End of layout drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 151, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1433, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2331), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/07_icon_BOTTLEROCK_Preferred_Shuttle_Bus_From_SA.png
try:
    _c7 = get_crop(7, 1344, 1091)
    canvas.paste(_c7, (48, 676), _c7)
except Exception:
    pass
layout["BOTTLEROCK_Preferred_Shut"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1236, 2331), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/10_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c10 = get_crop(10, 1344, 1001)
    canvas.paste(_c10, (48, 1815), _c10)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/11_icon_4.32.png
try:
    _c11 = get_crop(11, 128, 113)
    canvas.paste(_c11, (54, 116), _c11)
except Exception:
    pass
layout["4.32"] = [54, 116, 182, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 69, 62)
    canvas.paste(_c12, (307, 1), _c12)
except Exception:
    pass
layout["Search_forae"] = [307, 1, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/13_icon_4.32.png
try:
    _c13 = get_crop(13, 61, 63)
    canvas.paste(_c13, (181, 0), _c13)
except Exception:
    pass
layout["4.32"] = [181, 0, 242, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 62)
    canvas.paste(_c14, (246, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [246, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 94, 59)
    canvas.paste(_c16, (1208, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1208, 0, 1302, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/17_icon_4.32.png
try:
    _c17 = get_crop(17, 60, 64)
    canvas.paste(_c17, (115, 0), _c17)
except Exception:
    pass
layout["4.32"] = [115, 0, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 66, 59)
    canvas.paste(_c18, (1314, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1314, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/19_icon_To_bottlerock_2024.png
try:
    _c19 = get_crop(19, 1344, 1091)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["To_bottlerock_2024"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/20_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/21_icon_San_Francisco.png
try:
    _c21 = get_crop(21, 536, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 52, 61)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/23_icon_TI_00AM_PDT.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["TI:00AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/24_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 244, 67)
    canvas.paste(_c25, (84, 1659), _c25)
except Exception:
    pass
layout["Promoted"] = [84, 1659, 328, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/26_icon_4.32.png
try:
    _c26 = get_crop(26, 140, 63)
    canvas.paste(_c26, (8, 0), _c26)
except Exception:
    pass
layout["4.32"] = [8, 0, 148, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/27_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/28_icon_Sat_Jun_15.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Sat,_Jun_15"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 39, 60)
    canvas.paste(_c29, (1275, 0), _c29)
except Exception:
    pass
layout["icon_29"] = [1275, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/31_text_Westwood_Pickup_Location.png
try:
    _c31 = get_crop(31, 531, 56)
    canvas.paste(_c31, (92, 1600), _c31)
except Exception:
    pass
layout["Westwood_(Pickup_Location"] = [92, 1600, 623, 1656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/32_text_JINETEEN.png
try:
    _c32 = get_crop(32, 423, 113)
    canvas.paste(_c32, (523, 1825), _c32)
except Exception:
    pass
layout["JINETEEN"] = [523, 1825, 946, 1938]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/33_text_JUNE_15.png
try:
    _c33 = get_crop(33, 237, 81)
    canvas.paste(_c33, (1131, 1830), _c33)
except Exception:
    pass
layout["JUNE_15"] = [1131, 1830, 1368, 1911]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_02_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-4/34_text_Iortc.png
try:
    _c34 = get_crop(34, 109, 49)
    canvas.paste(_c34, (1158, 742), _c34)
except Exception:
    pass
layout["Iortc"] = [1158, 742, 1267, 791]
