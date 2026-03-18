# page_id: page_eventbrite_f56b9f3bf9ef483cbd9847af47d34434_04
# screenshot: 2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6.png
# step_index: 4/8
# task: Open Eventbrite. Look up "Gardening" events. Filter by events happening this week. Select the first event from the results. Follow the organizer and where is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas is provided (1440x2960), as well as draw, and fonts.
w, h = canvas.size

# Color palette (approximate to screenshot)
status_bar_color = (169, 169, 169)        # slightly darker grey for status bar
background_color = (250, 250, 252)        # near-white background
divider_color = (230, 231, 235)           # light divider
pill_color = (225, 242, 255)              # pale blue for filter pills
pill_border = (208, 228, 241)
event_image_bg = (245, 247, 250)          # very light image placeholder
event_card_bg = (255, 255, 255)           # card white
muted_bg = (244, 242, 247)                # pale lavender for big abstract placeholder
bottom_divider_color = (235, 235, 240)

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=background_color)

# Status bar (top ~60px)
status_h = 60
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)
# subtle dot to simulate notch area separation (non-icon)
draw.line([(0, status_h), (w, status_h)], fill=divider_color, width=1)

# Header area under status bar (~60-160)
header_top = status_h
header_h = 100
draw.rectangle([(0, header_top), (w, header_top + header_h)], fill=event_card_bg)
# bottom divider under header
draw.line([(24, header_top + header_h), (w - 24, header_top + header_h)], fill=divider_color, width=2)

# Filters/pills row area (approx center row around y ~400)
pills_y = 400
pill_height = 86
pill_radius = pill_height // 2
pill_specs = [
    (54, pills_y, 54 + 360, pills_y + pill_height),
    (440, pills_y, 440 + 420, pills_y + pill_height),
    (880, pills_y, 880 + 180, pills_y + pill_height),
    (1080, pills_y, 1080 + 240, pills_y + pill_height),
    (1340, pills_y, 1340 + 120, pills_y + pill_height),
]
for bbox in pill_specs:
    # draw pill background and a subtle border
    draw.rounded_rectangle(bbox, radius=pill_radius, fill=pill_color, outline=pill_border, width=2)

# Big results count/section divider area (thin line below pills)
line_y = pills_y + pill_height + 30
draw.line([(24, line_y), (w - 24, line_y)], fill=divider_color, width=2)

# First event card image placeholder (rounded)
card_x = 48
card_w = w - 2 * card_x  # 1344
img1_top = line_y + 30  # around 500
img1_h = 340
img1_bbox = [card_x, img1_top, card_x + card_w, img1_top + img1_h]
draw.rounded_rectangle(img1_bbox, radius=20, fill=event_image_bg)

# subtle shadow under first image
shadow_y1 = img1_top + img1_h
draw.rectangle([(card_x + 8, shadow_y1), (card_x + card_w - 8, shadow_y1 + 6)], fill=(230,230,232))

# First event card body (where title/time sit) - keep white
body1_top = img1_top + img1_h + 18
body1_h = 120
body1_bbox = [card_x, body1_top, card_x + card_w, body1_top + body1_h]
draw.rectangle(body1_bbox, fill=event_card_bg)
# subtle top divider
draw.line([(card_x + 12, body1_top), (card_x + card_w - 12, body1_top)], fill=divider_color, width=1)

# Separator before next event
sep_y = body1_top + body1_h + 26
draw.line([(24, sep_y), (w - 24, sep_y)], fill=divider_color, width=1)

# Second event large abstract image placeholder (rounded large pale shape)
img2_top = sep_y + 40
img2_h = 540
img2_bbox = [card_x, img2_top, card_x + card_w, img2_top + img2_h]
draw.rounded_rectangle(img2_bbox, radius=20, fill=muted_bg)

# Add a large curved "accent" inside second image to mimic abstract white shape
# We simulate it with a big white rounded ellipse clipped inside the image area by drawing an ellipse
ell_w = int(card_w * 1.2)
ell_h = int(img2_h * 0.9)
ell_x = card_x - int(card_w * 0.1)
ell_y = img2_top - int(img2_h * 0.15)
draw.ellipse([(ell_x, ell_y), (ell_x + ell_w, ell_y + ell_h)], fill=event_card_bg)

# subtle shadow under second image
shadow_y2 = img2_top + img2_h
draw.rectangle([(card_x + 8, shadow_y2), (card_x + card_w - 8, shadow_y2 + 6)], fill=(230,230,232))

# Second event card body area
body2_top = img2_top + img2_h + 22
body2_h = 160
body2_bbox = [card_x, body2_top, card_x + card_w, body2_top + body2_h]
draw.rectangle(body2_bbox, fill=event_card_bg)
draw.line([(card_x + 12, body2_top), (card_x + card_w - 12, body2_top)], fill=divider_color, width=1)

# Another subtle horizontal separator further down
draw.line([(24, body2_top + body2_h + 22), (w - 24, body2_top + body2_h + 22)], fill=divider_color, width=1)

# Bottom navigation bar (sticky)
bottom_h = 80
bottom_top = h - bottom_h
# top divider for nav bar
draw.line([(0, bottom_top), (w, bottom_top)], fill=bottom_divider_color, width=2)
draw.rectangle([(0, bottom_top), (w, h)], fill=event_card_bg)

# Add faint indicator dots above nav to simulate selection area (no icons)
sel_dot_x = int(w / 2)
draw.ellipse([(sel_dot_x - 18, bottom_top - 18), (sel_dot_x + 18, bottom_top + 18)], outline=(245, 128, 34), width=2)

# Final subtle left/right content separators near edges to emulate subtle layout lines
draw.line([(24, status_h + 6), (24, h - 120)], fill=divider_color, width=1)
draw.line([(w - 24, status_h + 6), (w - 24, h - 120)], fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2269), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2269, 1236, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/08_icon_Gardening.png
try:
    _c8 = get_crop(8, 1344, 191)
    canvas.paste(_c8, (48, 72), _c8)
except Exception:
    pass
layout["Gardening"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/09_icon_5.09.png
try:
    _c9 = get_crop(9, 114, 108)
    canvas.paste(_c9, (59, 117), _c9)
except Exception:
    pass
layout["5.09"] = [59, 117, 173, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/10_icon_Foo.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/11_icon_None.png
try:
    _c11 = get_crop(11, 353, 144)
    canvas.paste(_c11, (0, 259), _c11)
except Exception:
    pass
layout["None"] = [0, 259, 353, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/12_icon_Gardening.png
try:
    _c12 = get_crop(12, 68, 64)
    canvas.paste(_c12, (308, 0), _c12)
except Exception:
    pass
layout["Gardening"] = [308, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 104, 61)
    canvas.paste(_c13, (1206, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1206, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/14_icon_5.09.png
try:
    _c14 = get_crop(14, 58, 63)
    canvas.paste(_c14, (182, 0), _c14)
except Exception:
    pass
layout["5.09"] = [182, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 62)
    canvas.paste(_c15, (250, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [250, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/16_icon_5.09.png
try:
    _c16 = get_crop(16, 59, 65)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["5.09"] = [115, 0, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1236, 2269), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2269, 1380, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 60, 61)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/19_icon_Gardening_101_Online_Workshop.png
try:
    _c19 = get_crop(19, 1344, 1029)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["Gardening_101_Online_Work"] = [48, 676, 1392, 1705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/20_icon_Gardening.png
try:
    _c20 = get_crop(20, 49, 61)
    canvas.paste(_c20, (384, 3), _c20)
except Exception:
    pass
layout["Gardening"] = [384, 3, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/21_icon_2.30_PM_GMT_0I_00.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (576, 2804), _c21)
except Exception:
    pass
layout["2.30_PM_GMT+0I:00"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/22_icon_2.30_PM_GMT_0I_00.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["2.30_PM_GMT+0I:00"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/23_icon_More.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/24_icon_15_Prince_s_Gardens.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["15_Prince's_Gardens"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/25_icon_Free.png
try:
    _c25 = get_crop(25, 128, 76)
    canvas.paste(_c25, (91, 2446), _c25)
except Exception:
    pass
layout["Free"] = [91, 2446, 219, 2522]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/26_text_5.09.png
try:
    _c26 = get_crop(26, 91, 45)
    canvas.paste(_c26, (20, 15), _c26)
except Exception:
    pass
layout["5.09"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/27_text_9_127_events.png
try:
    _c27 = get_crop(27, 359, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["9,127_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/28_text_Gardening.png
try:
    _c28 = get_crop(28, 296, 76)
    canvas.paste(_c28, (91, 2533), _c28)
except Exception:
    pass
layout["Gardening"] = [91, 2533, 387, 2609]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/29_text_Wed_May_15.png
try:
    _c29 = get_crop(29, 257, 57)
    canvas.paste(_c29, (93, 2615), _c29)
except Exception:
    pass
layout["Wed,_May_15"] = [93, 2615, 350, 2672]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/30_text_2.30_PM_GMT_0I_00.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (288, 2804), _c30)
except Exception:
    pass
layout["2.30_PM_GMT+0I:00"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/31_text_15_Prince_s_Gardens.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (0, 2804), _c31)
except Exception:
    pass
layout["15_Prince's_Gardens"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_04_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-6/32_clickable_Event_s_image.png
try:
    _c32 = get_crop(32, 1344, 1029)
    canvas.paste(_c32, (48, 1753), _c32)
except Exception:
    pass
layout["Event's_image"] = [48, 1753, 1392, 2782]
