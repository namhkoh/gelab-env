# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_04
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6.png
# step_index: 4/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (canvas provided)
bg_color = (255, 255, 255)
status_color = (200, 200, 200)      # light gray for status bar
divider_color = (225, 225, 230)     # subtle dividers
pill_bg = (226, 243, 255)           # pale blue for filter pills
card_image_bg = (40, 40, 45)        # dark placeholder for image cards
card_banner_bg = (245, 246, 248)    # light neutral behind text blocks
bottom_bar_bg = (255, 255, 255)
shadow = (210, 210, 215)

# full background
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar (top)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)

# Thin inner highlight line under status bar
draw.line([(0, status_h), (1440, status_h)], fill=divider_color, width=1)

# Header / toolbar area
header_top = status_h
header_h = 94
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill=bg_color)
# header bottom divider
draw.line([(24, header_top + header_h), (1440 - 24, header_top + header_h)], fill=divider_color, width=2)

# Filter pill row background area (light background band)
filters_band_top = 378
filters_band_bottom = 528
draw.rectangle([(0, filters_band_top - 12), (1440, filters_band_bottom + 12)], fill=bg_color)

# Draw pill backgrounds (positions derived from detected elements)
pills = [
    (54, 410, 359, 103),
    (425, 410, 400, 103),
    (837, 410, 187, 103),
    (1036, 410, 241, 103),
    (1283, 406, 148, 110),
]
for (x, y, w, h) in pills:
    # slightly inset and rounded
    r = int(h / 2)
    draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=r, fill=pill_bg)

# "10,000 events" heading area (leave text out, draw subtle top padding line)
heading_top = 520
draw.line([(48, heading_top), (1440-48, heading_top)], fill=divider_color, width=1)

# First event card - image placeholder background (rounded rectangle)
first_img_x, first_img_y = 48, 676
first_img_w, first_img_h = 1344, 1175
# shadow
shadow_offset = 8
draw.rounded_rectangle(
    [(first_img_x + shadow_offset, first_img_y + shadow_offset),
     (first_img_x + first_img_w + shadow_offset, first_img_y + first_img_h + shadow_offset)],
    radius=28, fill=shadow
)
# image background
draw.rounded_rectangle(
    [(first_img_x, first_img_y), (first_img_x + first_img_w, first_img_y + first_img_h)],
    radius=24, fill=card_image_bg
)

# Card content band under the first image (text area background)
text_band_top = first_img_y + first_img_h + 18
text_band_h = 220
draw.rectangle([(48, text_band_top), (48 + 1344, text_band_top + text_band_h)], fill=card_banner_bg)
# subtle divider under text band
draw.line([(48, text_band_top + text_band_h + 8), (48 + 1344, text_band_top + text_band_h + 8)], fill=divider_color, width=1)

# Second event image placeholder (large promotional banner)
second_img_x, second_img_y = 48, 1899
second_img_w, second_img_h = 1344, 917
# shadow for second
draw.rounded_rectangle(
    [(second_img_x + shadow_offset, second_img_y + shadow_offset),
     (second_img_x + second_img_w + shadow_offset, second_img_y + second_img_h + shadow_offset)],
    radius=20, fill=shadow
)
draw.rounded_rectangle(
    [(second_img_x, second_img_y), (second_img_x + second_img_w, second_img_y + second_img_h)],
    radius=18, fill=(60, 60, 64)
)

# Subtle separator between list items
sep_y = second_img_y + second_img_h + 26
draw.line([(48, sep_y), (1440 - 48, sep_y)], fill=divider_color, width=1)

# Bottom navigation bar background and top divider
bottom_bar_top = 2796
draw.rectangle([(0, bottom_bar_top), (1440, 2960)], fill=bottom_bar_bg)
draw.line([(24, bottom_bar_top), (1440 - 24, bottom_bar_top)], fill=divider_color, width=2)

# Floating circular control backgrounds that belong as card decorations (but not icons)
# (Do not draw icon glyphs — only the plain circles behind where they appear)
floating_circles = [
    (1092, 1192, 144, 144),  # favorite button background area
    (1236, 1192, 144, 144),  # overflow/share button background area
    (1092, 2415, 144, 144),  # lower card small icons background area
    (1236, 2415, 144, 144),
]
for (cx, cy, w, h) in floating_circles:
    # center coords rounded circle
    r = min(w, h) // 2
    # draw a soft white disc with very light shadow (they will be overlaid by icons)
    draw.ellipse([(cx, cy), (cx + w, cy + h)], fill=(255, 255, 255))
    draw.ellipse([(cx + 2, cy + 2), (cx + w + 2, cy + h + 2)], outline=(240, 240, 240))

# finalize with a few subtle vertical paddings / guides (very light)
draw.line([(48, 150), (48, 2960 - 150)], fill=(250, 250, 250), width=1)
draw.line([(1440 - 48, 150), (1440 - 48, 2960 - 150)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 148, 110)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/07_icon_Icademy.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 2415), _c7)
except Exception:
    pass
layout["Icademy"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/08_icon_presehter.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["presehter"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/10_icon_5.25.png
try:
    _c10 = get_crop(10, 124, 112)
    canvas.paste(_c10, (54, 114), _c10)
except Exception:
    pass
layout["5.25"] = [54, 114, 178, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/11_icon_Tech.png
try:
    _c11 = get_crop(11, 69, 65)
    canvas.paste(_c11, (307, 0), _c11)
except Exception:
    pass
layout["Tech"] = [307, 0, 376, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/12_icon_5.25.png
try:
    _c12 = get_crop(12, 61, 65)
    canvas.paste(_c12, (181, 0), _c12)
except Exception:
    pass
layout["5.25"] = [181, 0, 242, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/13_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c13 = get_crop(13, 1344, 1175)
    canvas.paste(_c13, (48, 676), _c13)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/14_icon_5.25.png
try:
    _c14 = get_crop(14, 61, 66)
    canvas.paste(_c14, (114, 0), _c14)
except Exception:
    pass
layout["5.25"] = [114, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/15_icon_Tech.png
try:
    _c15 = get_crop(15, 54, 66)
    canvas.paste(_c15, (246, 0), _c15)
except Exception:
    pass
layout["Tech"] = [246, 0, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/16_icon_Online.png
try:
    _c16 = get_crop(16, 377, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 102, 61)
    canvas.paste(_c17, (1206, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1206, 0, 1308, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 60)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1380, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/19_icon_And_Runnina_Youir_LLC.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["And_Runnina_Youir_LLC"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/20_icon_Tech.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Tech"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/21_icon_Promoted.png
try:
    _c21 = get_crop(21, 250, 67)
    canvas.paste(_c21, (78, 1742), _c21)
except Exception:
    pass
layout["Promoted"] = [78, 1742, 328, 1809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/22_icon_Icademy.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["Icademy"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/23_icon_Free.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Free"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 52, 62)
    canvas.paste(_c24, (383, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [383, 2, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/25_icon_Everything_You_Need_To_Know_About_Starti.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Everything_You_Need_To_Kn"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/26_icon_Everything_You_Need_To_Know_About_Starti.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Everything_You_Need_To_Kn"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/27_icon_Tickets.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/28_text_5.25.png
try:
    _c28 = get_crop(28, 92, 43)
    canvas.paste(_c28, (22, 17), _c28)
except Exception:
    pass
layout["5.25"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/29_text_10_000_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/30_text_Online.png
try:
    _c30 = get_crop(30, 126, 43)
    canvas.paste(_c30, (94, 1689), _c30)
except Exception:
    pass
layout["Online"] = [94, 1689, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/31_text_INTERACTIVE.png
try:
    _c31 = get_crop(31, 209, 38)
    canvas.paste(_c31, (66, 1916), _c31)
except Exception:
    pass
layout["INTERACTIVE"] = [66, 1916, 275, 1954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/32_text_MORKSHOP.png
try:
    _c32 = get_crop(32, 191, 36)
    canvas.paste(_c32, (68, 1955), _c32)
except Exception:
    pass
layout["MORKSHOP"] = [68, 1955, 259, 1991]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/33_text_EVERYTHING_YOU.png
try:
    _c33 = get_crop(33, 415, 52)
    canvas.paste(_c33, (65, 2021), _c33)
except Exception:
    pass
layout["EVERYTHING_YOU"] = [65, 2021, 480, 2073]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/34_text_NEED_TO_KNOW.png
try:
    _c34 = get_crop(34, 371, 52)
    canvas.paste(_c34, (67, 2081), _c34)
except Exception:
    pass
layout["NEED_TO_KNOW"] = [67, 2081, 438, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/35_text_ABOUT_STARTING.png
try:
    _c35 = get_crop(35, 408, 51)
    canvas.paste(_c35, (67, 2142), _c35)
except Exception:
    pass
layout["ABOUT_STARTING"] = [67, 2142, 475, 2193]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/36_text_AND_RUNNING_YOUR_LLC.png
try:
    _c36 = get_crop(36, 1344, 917)
    canvas.paste(_c36, (48, 1899), _c36)
except Exception:
    pass
layout["AND_RUNNING_YOUR_LLC"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/37_text_presehter.png
try:
    _c37 = get_crop(37, 186, 36)
    canvas.paste(_c37, (1139, 2249), _c37)
except Exception:
    pass
layout["presehter"] = [1139, 2249, 1325, 2285]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/38_text_Diecuntiani.png
try:
    _c38 = get_crop(38, 147, 29)
    canvas.paste(_c38, (125, 2285), _c38)
except Exception:
    pass
layout["Diecuntiani"] = [125, 2285, 272, 2314]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/39_text_Insighta.png
try:
    _c39 = get_crop(39, 105, 36)
    canvas.paste(_c39, (293, 2284), _c39)
except Exception:
    pass
layout["Insighta"] = [293, 2284, 398, 2320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/40_text_6_uC_orolee.png
try:
    _c40 = get_crop(40, 171, 30)
    canvas.paste(_c40, (76, 2326), _c40)
except Exception:
    pass
layout["6_uC+orolee"] = [76, 2326, 247, 2356]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/41_text_Jotlmt_phna.png
try:
    _c41 = get_crop(41, 112, 27)
    canvas.paste(_c41, (377, 2331), _c41)
except Exception:
    pass
layout["Jotlmt_phna"] = [377, 2331, 489, 2358]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/42_text_Voit.png
try:
    _c42 = get_crop(42, 82, 27)
    canvas.paste(_c42, (152, 2372), _c42)
except Exception:
    pass
layout["Voit"] = [152, 2372, 234, 2399]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/43_text_Cemn_ansa_Stateg_1.png
try:
    _c43 = get_crop(43, 212, 31)
    canvas.paste(_c43, (380, 2371), _c43)
except Exception:
    pass
layout["Cemn_ansa_Stateg*1"] = [380, 2371, 592, 2402]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/44_text_LIVE_08A.png
try:
    _c44 = get_crop(44, 1344, 917)
    canvas.paste(_c44, (48, 1899), _c44)
except Exception:
    pass
layout["LIVE_08A"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/45_text_This_wortshop_I_Ideal_forthoso_coralderl.png
try:
    _c45 = get_crop(45, 1344, 917)
    canvas.paste(_c45, (48, 1899), _c45)
except Exception:
    pass
layout["This_wortshop_I_Ideal_for"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/46_text_Mytech.png
try:
    _c46 = get_crop(46, 147, 54)
    canvas.paste(_c46, (923, 2454), _c46)
except Exception:
    pass
layout["Mytech"] = [923, 2454, 1070, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/47_text_compliancu_and_ollectiva_manogemunt_stra.png
try:
    _c47 = get_crop(47, 1344, 917)
    canvas.paste(_c47, (48, 1899), _c47)
except Exception:
    pass
layout["compliancu_and_ollectiva_"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/48_text_Free.png
try:
    _c48 = get_crop(48, 80, 39)
    canvas.paste(_c48, (117, 2614), _c48)
except Exception:
    pass
layout["Free"] = [117, 2614, 197, 2653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_04_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-6/49_text_Everything_You_Need_To_Know_About_Starti.png
try:
    _c49 = get_crop(49, 1344, 917)
    canvas.paste(_c49, (48, 1899), _c49)
except Exception:
    pass
layout["Everything_You_Need_To_Kn"] = [48, 1899, 1392, 2816]
