# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_10
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12.png
# step_index: 10/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 canvas using PIL (canvas & draw provided)

# Overall background (slightly off-white to match screenshot)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFC")

# Status bar area (top ~50px-80px) - darker bar to match device status area
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#9EA3A8")

# Header / Toolbar area (below status bar)
header_top = status_h
header_bottom = 200
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# subtle bottom divider for header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#E6E7EA", width=2)

# Search/title row shadow hint (very subtle)
draw.line([(24, header_bottom+4), (1440-24, header_bottom+4)], fill="#F3F4F6", width=1)

# Filter band background (light blue rounded band behind filter pills)
filters_top = 256
filters_bottom = 360
filters_margin = 24
draw.rounded_rectangle(
    [(filters_margin, filters_top), (1440 - filters_margin, filters_bottom)],
    radius=48,
    fill="#EAF5FF",
    outline=None
)

# Divider under filters
draw.line([(24, filters_bottom + 12), (1440 - 24, filters_bottom + 12)], fill="#ECEEF2", width=1)

# First event card container (white rounded card with subtle shadow)
card1_x0 = 24
card1_x1 = 1440 - 24
card1_y0 = 420
card1_y1 = 980
# shadow (subtle)
draw.rounded_rectangle(
    [(card1_x0+6, card1_y0+8), (card1_x1+6, card1_y1+8)],
    radius=24,
    fill="#F5F6F8"
)
# card surface
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=20,
    fill="#FFFFFF"
)
# thin separator line inside card (under header area within card)
draw.line([(card1_x0+24, card1_y0+110), (card1_x1-24, card1_y0+110)], fill="#F0F1F4", width=1)

# Decorative slim image banner at top of first card (colorful strip, not an icon)
banner_y = card1_y0 + 10
draw.rectangle([(card1_x0+24, banner_y), (card1_x1-24, banner_y+18)], fill="#E7CFE8")

# Large image/content area placeholder inside first card (light neutral shape)
image1_x0 = card1_x0 + 24
image1_x1 = card1_x1 - 24
image1_y0 = card1_y0 + 140
image1_y1 = card1_y0 + 420
draw.rounded_rectangle(
    [(image1_x0, image1_y0), (image1_x1, image1_y1)],
    radius=18,
    fill="#F2F1F4"
)

# Small circular white action backgrounds at bottom-right of image (background only)
action_r = 44
action_gap = 28
# rightmost
draw.ellipse([(image1_x1 - action_gap - action_r*2, image1_y1 - action_gap - action_r*2),
              (image1_x1 - action_gap, image1_y1 - action_gap)], fill="#FFFFFF")
# second
draw.ellipse([(image1_x1 - action_gap*3 - action_r*2, image1_y1 - action_gap - action_r*2),
              (image1_x1 - action_gap*2 - action_r*0, image1_y1 - action_gap)], fill="#FFFFFF")

# Separator between first card and next content
sep_y = card1_y1 + 24
draw.line([(24, sep_y), (1440-24, sep_y)], fill="#F0F1F4", width=1)

# Second event card container (white rounded)
card2_y0 = sep_y + 18
card2_y1 = 2000
draw.rounded_rectangle(
    [(card1_x0, card2_y0), (card1_x1, card2_y1)],
    radius=20,
    fill="#FFFFFF"
)
# subtle divider near top of second card
draw.line([(card1_x0+24, card2_y0+120), (card1_x1-24, card2_y0+120)], fill="#F0F1F4", width=1)

# Large image/banner area for the second card (bright colored background area)
banner2_x0 = card1_x0 + 20
banner2_x1 = card1_x1 - 20
banner2_y0 = card2_y0 + 240
banner2_y1 = banner2_y0 + 520
draw.rounded_rectangle(
    [(banner2_x0, banner2_y0), (banner2_x1, banner2_y1)],
    radius=18,
    fill="#FFD842"
)

# Content area white band below second banner
content_band_y0 = banner2_y1 + 28
content_band_y1 = content_band_y0 + 140
draw.rectangle([(24, content_band_y0), (1440-24, content_band_y1)], fill="#FFFFFF")
draw.line([(24, content_band_y1), (1440-24, content_band_y1)], fill="#ECEFF2", width=1)

# Additional subtle horizontal separators at logical section breaks
for y in (card2_y0 + 360, card2_y0 + 720, card2_y0 + 1100):
    draw.line([(24, y), (1440-24, y)], fill="#F5F6F8", width=1)

# Bottom navigation bar background
nav_h = 100
nav_top = 2960 - nav_h
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
# top divider for nav bar
draw.line([(24, nav_top), (1440-24, nav_top)], fill="#E6E7EA", width=2)

# Subtle floating handle above nav (common on mobile UIs)
handle_w = 120
handle_h = 6
handle_x0 = (1440 - handle_w) // 2
handle_y0 = nav_top - 20
draw.rounded_rectangle([(handle_x0, handle_y0), (handle_x0 + handle_w, handle_y0 + handle_h)], radius=3, fill="#E7E9EB")

# Final overall subtle vignette lines to imply card edges (very light)
draw.line([(24, 420), (24, card2_y1)], fill="#FAFAFB", width=1)
draw.line([(1440-24, 420), (1440-24, card2_y1)], fill="#FAFAFB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/00_icon_Music.png
try:
    _c0 = get_crop(0, 197, 112)
    canvas.paste(_c0, (829, 405), _c0)
except Exception:
    pass
layout["Music"] = [829, 405, 1026, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/01_icon_Business.png
try:
    _c1 = get_crop(1, 250, 110)
    canvas.paste(_c1, (1029, 406), _c1)
except Exception:
    pass
layout["Business"] = [1029, 406, 1279, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 408, 114)
    canvas.paste(_c2, (418, 405), _c2)
except Exception:
    pass
layout["Anytime"] = [418, 405, 826, 519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/03_icon_Filters.png
try:
    _c3 = get_crop(3, 434, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["Filters"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/04_icon_Foo.png
try:
    _c4 = get_crop(4, 154, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1595), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1595, 1236, 1739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/06_icon_Sales_ended.png
try:
    _c6 = get_crop(6, 254, 81)
    canvas.paste(_c6, (90, 1770), _c6)
except Exception:
    pass
layout["Sales_ended"] = [90, 1770, 344, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/07_icon_Foo.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 96), _c7)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 65)
    canvas.paste(_c8, (1152, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1152, 1, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 57, 61)
    canvas.paste(_c9, (246, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [246, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/10_icon_9.45.png
try:
    _c10 = get_crop(10, 123, 116)
    canvas.paste(_c10, (55, 113), _c10)
except Exception:
    pass
layout["9.45"] = [55, 113, 178, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1236, 1595), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1595, 1380, 1739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/12_icon_The_Original_E._Village_Food_Drinks_Cult.png
try:
    _c12 = get_crop(12, 1344, 1108)
    canvas.paste(_c12, (48, 1079), _c12)
except Exception:
    pass
layout["The_Original_E._Village_F"] = [48, 1079, 1392, 2187]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 95, 64)
    canvas.paste(_c13, (1211, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1211, 0, 1306, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/14_icon_9.45.png
try:
    _c14 = get_crop(14, 55, 62)
    canvas.paste(_c14, (182, 0), _c14)
except Exception:
    pass
layout["9.45"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 57, 63)
    canvas.paste(_c15, (313, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [313, 1, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/16_icon_New_York.png
try:
    _c16 = get_crop(16, 434, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 54, 61)
    canvas.paste(_c17, (1319, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 1, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/18_icon_Food_Drink.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/19_icon_9.45.png
try:
    _c19 = get_crop(19, 55, 64)
    canvas.paste(_c19, (115, 0), _c19)
except Exception:
    pass
layout["9.45"] = [115, 0, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/20_icon_Aneese.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["Aneese"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/21_icon_Food_Drink.png
try:
    _c21 = get_crop(21, 48, 61)
    canvas.paste(_c21, (383, 2), _c21)
except Exception:
    pass
layout["Food_&_Drink"] = [383, 2, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/22_icon_230_Fifth_Rooftop_Bar.png
try:
    _c22 = get_crop(22, 45, 66)
    canvas.paste(_c22, (283, 925), _c22)
except Exception:
    pass
layout["230_Fifth_Rooftop_Bar"] = [283, 925, 328, 991]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/23_icon_Aneese.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Aneese"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/24_icon_Aneese.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Aneese"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/25_icon_Tickets.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/27_icon_6.30PM-9_30PM.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["6.30PM-9:30PM"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/28_icon_The_Original_E._Village_Food_Drinks_Cult.png
try:
    _c28 = get_crop(28, 1344, 1108)
    canvas.paste(_c28, (48, 1079), _c28)
except Exception:
    pass
layout["The_Original_E._Village_F"] = [48, 1079, 1392, 2187]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/29_text_9.45.png
try:
    _c29 = get_crop(29, 94, 43)
    canvas.paste(_c29, (20, 15), _c29)
except Exception:
    pass
layout["9.45"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/30_text_Ticket_sales_end_soon.png
try:
    _c30 = get_crop(30, 415, 51)
    canvas.paste(_c30, (125, 629), _c30)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [125, 629, 540, 680]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/31_text_The_HOLI-CON_Ball_Festival_230_Sth_Rooft.png
try:
    _c31 = get_crop(31, 1344, 506)
    canvas.paste(_c31, (48, 525), _c31)
except Exception:
    pass
layout["The_HOLI-CON_Ball_Festiva"] = [48, 525, 1392, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/32_text_Sat_Mar_23.png
try:
    _c32 = get_crop(32, 235, 53)
    canvas.paste(_c32, (90, 798), _c32)
except Exception:
    pass
layout["Sat,_Mar_23"] = [90, 798, 325, 851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/33_text_1.55_PM_EDT.png
try:
    _c33 = get_crop(33, 250, 53)
    canvas.paste(_c33, (345, 796), _c33)
except Exception:
    pass
layout["1.55_PM_EDT"] = [345, 796, 595, 849]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/34_text_230_Fifth_Rooftop_Bar.png
try:
    _c34 = get_crop(34, 406, 55)
    canvas.paste(_c34, (90, 865), _c34)
except Exception:
    pass
layout["230_Fifth_Rooftop_Bar"] = [90, 865, 496, 920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/35_text_Thursday_April_25th.png
try:
    _c35 = get_crop(35, 1344, 581)
    canvas.paste(_c35, (48, 2235), _c35)
except Exception:
    pass
layout["Thursday_April_25th,"] = [48, 2235, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/36_text_2024.png
try:
    _c36 = get_crop(36, 157, 66)
    canvas.paste(_c36, (1217, 2268), _c36)
except Exception:
    pass
layout["2024"] = [1217, 2268, 1374, 2334]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/37_text_6.30PM-9_30PM.png
try:
    _c37 = get_crop(37, 431, 63)
    canvas.paste(_c37, (806, 2356), _c37)
except Exception:
    pass
layout["6.30PM-9:30PM"] = [806, 2356, 1237, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_10_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-12/38_text_The.png
try:
    _c38 = get_crop(38, 223, 136)
    canvas.paste(_c38, (110, 2313), _c38)
except Exception:
    pass
layout["The"] = [110, 2313, 333, 2449]
